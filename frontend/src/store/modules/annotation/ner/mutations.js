import { Notify } from "quasar";

export default {
  updateResult(state, payload) {
    let autoEnt = [];
    let validEnt = state.ents.concat(["null"]);

    payload.forEach(({ token, label, iob, confidence }) => {
      if (validEnt.includes(label) && confidence >= 0.2) {
        autoEnt.push({
          token: token,
          label: label,
          iob: iob,
          confidence: confidence,
        });
      } else {
        autoEnt.push({
          token: token,
          label: "null",
          iob: 2,
          confidence: confidence,
        });
      }
    });
    state.result = autoEnt;
  },
  updateCurEnt(state, payload) {
    state.curEnt = payload;
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
  updateDataset(state, payload) {
    state.dataset = payload;
  },
  updateAL(state, payload) {
    state.al = payload;
  },
  updateEarlyPhaseOn(state, payload) {
    state.earlyPhaseOn = payload;
  },
  updateActiveLearningOn(state, payload) {
    state.activeLearningOn = payload;
  },
  updateShowConfidence(state, payload) {
    state.showConfidence = payload;
  },
  updateAutoSuggestStrength(state, payload) {
    state.autoSuggestStrength = payload;
  },
  updateTradeoff(state, payload) {
    state.tradeoff = payload;
  },
  updateStartIndex(state, payload) {
    state.startIndex = payload;
  },
  updateStartAnnotation(state, payload) {
    state.startAnnotation = payload;
  },
  updateDataIndex(state, payload) {
    state.dataIndex = payload;
  },

  addHistory(state, payload) {
    state.historyResult.push(JSON.stringify(state.result));
    state.historyIndex.push(state.dataIndex);
    state.historyCurInd = state.historyIndex.length - 1;
  },

  saveHistory(state, payload) {
    state.historyResult[state.historyCurInd] = JSON.stringify(state.result);
  },

  retrieveHistoryPrev(state, payload) {
    const ind =
      (state.historyCurInd + state.historyIndex.length - 1) %
      state.historyIndex.length;
    state.historyResult[state.historyCurInd] = JSON.stringify(state.result);
    state.result = JSON.parse(state.historyResult[ind]);
    state.dataIndex = state.historyIndex[ind];
    state.historyCurInd = ind;
  },

  retrieveHistoryNext(state, payload) {
    const ind =
      (state.historyCurInd + state.historyIndex.length + 1) %
      state.historyIndex.length;
    state.historyResult[state.historyCurInd] = JSON.stringify(state.result);
    state.result = JSON.parse(state.historyResult[ind]);
    state.dataIndex = state.historyIndex[ind];
    state.historyCurInd = ind;
  },

  toggleLabel(state, payload) {
    const idList = payload.split(",").map(Number);
    if (idList.length === 1) {
      const id = idList;
      if (state.result[id]["label"] === "null") {
        state.result[id] = {
          token: state.result[id]["token"],
          confidence: 1.0,
          label: state.curEnt,
          iob: 3,
        };
      } else {
        state.result[id] = {
          token: state.result[id]["token"],
          confidence: state.result[id]["confidence"],
          label: "null",
          iob: 2,
        };
      }
    } else {
      idList.forEach((id) => {
        state.result[id] = {
          token: state.result[id]["token"],
          confidence: state.result[id]["confidence"],
          label: "null",
          iob: 2,
        };
      });
    }
  },

  addLabel(state, payload) {
    const startIndex = parseInt(state.startIndex);
    const endIndex = parseInt(payload.endIndex);

    let id = null;

    for (id = startIndex; id <= endIndex; id++) {
      if (state.result[id]["iob"] !== 2) {
        Notify.create({
          color: "negative",
          position: "top",
          message: "Invalid selection! Label Overlap!",
          icon: "report_problem",
        });
        console.log(
          "Invalid selection! Label Overlap! ",
          `Token: ${state.result[id]["token"]}, Label: ${state.result[id]["label"]}, IOB: ${state.result[id]["iob"]}`
        );
        return;
      }
    }

    for (id = startIndex; id <= endIndex; id++) {
      if (id === startIndex) {
        state.result[id] = {
          token: state.result[id]["token"],
          confidence: state.result[id]["confidence"],
          label: state.curEnt,
          iob: 3,
        };
      } else {
        state.result[id] = {
          token: state.result[id]["token"],
          confidence: state.result[id]["confidence"],
          label: state.curEnt,
          iob: 1,
        };
      }
    }
  },
};
