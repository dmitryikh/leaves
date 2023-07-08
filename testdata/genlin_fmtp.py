import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import datasets
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split


def data() -> pd.DataFrame:
    """ Fetch and preprocess the French Motor Third-Party Liability Claims dataset
    """

    # Policy definitions
    df_freq = datasets.fetch_openml(data_id=41214, as_frame=True, parser="pandas").data

    # Claims
    df_sev = datasets.fetch_openml(data_id=41215, as_frame=True, parser="pandas").data

    df_freq = (df_freq
               .assign(IDpol=lambda x: x['IDpol'].astype(int))
               .set_index("IDpol"))

    # Sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = (df_freq
          .join(df_sev, how="left")
          .assign(ClaimAmount=lambda x: x['ClaimAmount'].fillna(0).clip(upper=200000),
                  ClaimNb=lambda x: x["ClaimNb"].clip(upper=4),
                  Exposure=lambda x: x["Exposure"].clip(upper=1)))

    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0

    # Insurances companies are interested in modeling the Pure Premium, that is
    # the expected total claim amount per unit of exposure for each policyholder
    # in their portfolio:
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]

    # This can be indirectly approximated by a 2-step modeling: the product of the
    # Frequency times the average claim amount per claim:
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)

    return df


def train_model(objective: str, df_tr: pd.DataFrame, df_te: pd.DataFrame):
    """ Train different models based on the objective """

    assert objective in ('poisson', 'gamma', 'tweedie')

    target, weight = None, None
    match objective:
        case 'poisson':
            target, weight = 'Frequency', 'Exposure'
        case 'gamma':
            target, weight = 'AvgClaimAmount', 'ClaimNb'
        case 'tweedie':
            target, weight = 'PurePremium', 'Exposure'

    parameters = dict(
        objective=objective,
        learning_rate=.05,
        num_leaves=2 ** 5,
        min_data_in_leaf=10)

    d_train = lgb.Dataset(
        data=df_tr[feats],
        label=df_tr[target].values,
        weight=df_tr[weight].values,
        feature_name=feats,
        categorical_feature=cat_feats)

    regr = lgb.train(
        params=parameters,
        train_set=d_train,
        num_boost_round=100,
        feature_name=feats,
        categorical_feature=cat_feats)

    y_pred = regr.predict(df_te[feats])

    regr.save_model(f'genlin_fmtp_{objective}_{target}.model')
    np.savetxt(f'genlin_fmtp_{objective}_{target}_true_predictions.txt', y_pred, delimiter='\t')
    np.savetxt(f'genlin_fmtp_{objective}_{target}_features.tsv', df_te[feats], delimiter='\t', fmt='%d')


df = data()
df_train, df_test = train_test_split(df, test_size=1_000, random_state=0)

cat_feats = ['VehBrand', 'VehPower', 'VehGas', 'Region', 'Area']
num_feats = ['VehAge', 'DrivAge', 'BonusMalus', 'Density']
feats = cat_feats + num_feats

oe = OrdinalEncoder()
oe.fit(df_train[cat_feats])

# The continuous features are integers
df_train[cat_feats] = oe.transform(df_train[cat_feats]).astype(int)
df_train[num_feats] = df_train[num_feats].astype(int)

df_test[cat_feats] = oe.transform(df_test[cat_feats]).astype(int)
df_test[num_feats] = df_test[num_feats].astype(int)

mask_gamma = lambda x: x['ClaimAmount'] > 0

train_model(objective='poisson', df_tr=df_train, df_te=df_test)
train_model(objective='gamma', df_tr=df_train[mask_gamma], df_te=df_test[mask_gamma])
train_model(objective='tweedie', df_tr=df_train, df_te=df_test)
