package leaves

func GetNLeaves(trees []lgTree) []int {
	result := make([]int, len(trees))
	for idx, tree := range trees {
		result[idx] = tree.nLeaves()
	}

	return result
}