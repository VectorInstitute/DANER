import { Notify } from "quasar";
import { api } from "boot/axios";

export default {
  async getDataset(context, payload) {
    let destination = "";
    if (context.state.earlyPhaseOn) {
      destination = "/annotation_scratch";
    } else {
      destination = "/annotation";
    }
    const response = await api.get(destination);
    // console.log(response);

    if (response.status !== 200) {
      Notify.create({
        color: "negative",
        position: "top",
        message: "Loading failed",
        icon: "report_problem",
      });
    } else {
      context.commit("updateResult", response.data["ents"]);
      context.commit("updateDataIndex", response.data["index"]);
      context.commit("addHistory");
    }
  },

  async updateDataset(context, payload) {
    let conf = context.state.autoSuggestStrength / 100;
    let validEnt = context.state.ents.concat(["null"]);
    let autoEnt = [];

    context.state.result.forEach(({ token, label, iob, confidence }) => {
      if (validEnt.includes(label) && confidence >= conf) {
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

    const label = JSON.parse(JSON.stringify(context.state.result));

    const queryData = {
      crossdomain: true,
      index: context.state.dataIndex,
      annotator: context.state.annotator,
      label: JSON.stringify(label),
    };

    let destination = "";
    if (context.state.earlyPhaseOn) {
      destination = "/annotation_scratch";
    } else {
      destination = "/annotation";
    }
    const response = await api.put(destination, queryData);

    if (response.status !== 201) {
      Notify.create({
        color: "negative",
        position: "top",
        message: "Loading failed",
        icon: "report_problem",
      });
    } else {
      context.commit("saveHistory");
      context.dispatch("getDataset");
    }
  },
};
