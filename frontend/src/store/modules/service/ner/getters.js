export default {
  getAllEnts(state) {
    return state.ALL_ENTS.map((ele) => ele.val);
  },
  getDefaultEnts(state) {
    return state.DEFAULT_ENTS;
  },
  getDefaultModel(state) {
    return state.DEFAULT_MODEL;
  },
};
