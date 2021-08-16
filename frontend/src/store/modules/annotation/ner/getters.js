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
  getEntList(state) {
    return state.ALL_ENTS.filter(({ val }) => state.ents.includes(val));
  },

  getContentList(state) {
    let tmpColor = null;
    let tmpLabel = null;
    let childNodes = [];
    let entityToken = [];
    let entityId = [];
    let entconf = [];

    let conf = state.autoSuggestStrength / 100;
    const average = (array) => array.reduce((a, b) => a + b) / array.length;
    

    if (state.result.length > 0) {
      state.result.forEach(({ token, label, iob, confidence }, index) => {
        if (state.ents.includes(label) && confidence >= conf) {
          if (iob === 3) {
            if (entityToken.length > 0) {
              childNodes.push({
                token: entityToken.join(" "),
                label: tmpLabel,
                color: tmpColor,
                id: entityId,
                confidence: average(entconf),
              });
              entityToken = [];
              entityId = [];
              entconf = [];
            }
            tmpColor = state.ALL_ENTS.find(({ val }) => val === label).color;
            tmpLabel = label;
            // console.log(tmpColor);
            // console.log(tmpLabel);
          }
          entityToken.push(token);
          entityId.push(index);
          entconf.push(confidence);
        } else {
          if (entityToken.length > 0) {
            childNodes.push({
              token: entityToken.join(" "),
              label: tmpLabel,
              color: tmpColor,
              id: entityId,
              confidence: average(entconf),
            });
            entityToken = [];
            entityId = [];
            entconf = [];
          }
          childNodes.push({
            token: token,
            label: "",
            color: "",
            id: index,
            confidence: confidence,
          });
        }
      });
      if (entityToken.length > 0) {
        childNodes.push({
          token: entityToken.join(" "),
          label: tmpLabel,
          color: tmpColor,
          id: entityId,
          confidence: average(entconf),
        });
        entityToken = [];
        entityId = [];
        entconf = [];
      }
    }
    return childNodes;
  },
};
