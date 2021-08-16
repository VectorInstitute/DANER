import { Notify } from "quasar";
import { api } from "boot/axios";

export default {
  async analyzeText(context, payload) {
    const queryData = {
      crossdomain: true,
      model: context.state.model,
      text: payload.textInput,
      mode: "char",
    };

    const response = await api.put("/spacy", queryData);
    console.log(response);
    
    if (response.status !== 201) {
      Notify.create({
        color: "negative",
        position: "top",
        message: "Loading failed",
        icon: "report_problem",
      });
    } else {
      context.commit("updateResult", response.data["ents"]);
    }
  },
};
