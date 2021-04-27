package xgjson

type Objective struct {
	Name         string       `json:"name"`
	RegLossParam RegLossParam `json:"reg_loss_param"`
}

type RegLossParam struct {
	ScalePosWeight string `json:"scale_pos_weight"`
}
