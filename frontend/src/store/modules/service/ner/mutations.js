export default {
  updateResult(state, payload) {
    state.result = payload;
  },
  updateText(state, payload) {
    state.text = payload;
    state.result = [];
  },
  updateEnts(state, payload) {
    state.ents = payload;
  },
  updateModel(state, payload) {
    state.model = payload;
  },
};
