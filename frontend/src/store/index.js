import { store } from "quasar/wrappers";
import { createStore } from "vuex";
import { api } from "boot/axios";

import serviceNer from "./modules/service/ner/index";
import annotationNer from "./modules/annotation/ner/index";

/*
 * If not building with SSR mode, you can
 * directly export the Store instantiation;
 *
 * The function below can be async too; either use
 * async/await or return a Promise which resolves
 * with the Store instance.
 */

export default store(function (/* { ssrContext } */) {
  const Store = createStore({
    modules: {
      serviceNer,
      annotationNer,
    },
    state() {
      return {
        baseURL: "http://172.17.8.59:5000",
      };
    },
    mutations: {
      updateBaseURL(state, payload) {
        state.baseURL = payload;
        api.defaults.baseURL = payload;
      },
    },
    // enable strict mode (adds overhead!)
    // for dev mode and --debug builds only
    strict: process.env.DEBUGGING,
  });

  return Store;
});
