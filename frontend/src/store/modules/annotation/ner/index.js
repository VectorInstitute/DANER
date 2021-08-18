import getters from "./getters";
import mutations from "./mutations";
import actions from "./actions";

export default {
  namespaced: true,
  state() {
    return {
      // Glossary: https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
      // A small subset of ENTs
      ALL_ENTS: [
        { val: "ORG", label: "ORG", color: "cyan-3" },
        { val: "LOC", label: "LOC", color: "orange-4" },
        { val: "PER", label: "PER", color: "pink-3" },
        { val: "MISC", label: "MISC", color: "purple-5" },
        { val: "PRODUCT", label: "PRODUCT", color: "green-4" },
        { val: "DATE", label: "DATE", color: "teal-3" },
        { val: "TIME", label: "TIME", color: "teal-3" },
        { val: "MONEY", label: "MONEY", color: "lime-7" },
        { val: "QUANTITY", label: "QUANTITY", color: "lime-7" },
      ],

      // ALL_ENTS: [
      //   { val: "ORG", label: "ORG", color: "cyan-3" },
      //   { val: "LOC", label: "LOC", color: "orange-4" },
      //   { val: "PERSON", label: "PERSON", color: "pink-3" },
      //   { val: "MISC", label: "MISC", color: "purple-5" },
      //   { val: "PRODUCT", label: "PRODUCT", color: "green-4" },
      //   { val: "GPE", label: "GPE", color: "orange-4" },
      //   { val: "NORP", label: "NORP", color: "deep-purple-5" },
      //   { val: "FACILITY", label: "FACILITY", color: "cyan-8" },
      //   { val: "EVENT", label: "EVENT", color: "indigo-3" },
      //   { val: "LAW", label: "LAW", color: "red-3" },
      //   { val: "LANGUAGE", label: "LANGUAGE", color: "red-3" },
      //   { val: "WORK_OF_ART", label: "ART", color: "purple-3" },
      //   { val: "DATE", label: "DATE", color: "teal-3" },
      //   { val: "TIME", label: "TIME", color: "teal-3" },
      //   { val: "MONEY", label: "MONEY", color: "lime-7" },
      //   { val: "QUANTITY", label: "QUANTITY", color: "lime-7" },
      //   { val: "ORDINAL", label: "ORDINAL", color: "lime-7" },
      //   { val: "CARDINAL", label: "CARDINAL", color: "lime-7" },
      //   { val: "PERCENT", label: "PERCENT", color: "lime-7" },
      // ],
      ALL_MODELS: [
        "hf_distillbert",
        "en_core_web_sm",
        "en_core_web_md",
        "en_core_web_lg",
        "en_core_web_trf",
      ],
      ALL_DATASETS: [
        "CONLL2003",
      ],
      ALL_ALS: [
        "MNLP", // Maximum Normalized Log-Probability
        "LC" // Least Confidence
      ],

      DEFAULT_ENTS: ["ORG", "PER", "LOC", "MISC"],
      DEFAULT_MODEL: "hf_distillbert",

      model: "hf_distillbert",
      dataset: "CONLL2003",
      al: "MNLP",
      annotator: "admin",
      ents: ["ORG", "PER", "LOC", "MISC"],
      earlyPhaseOn: true,
      activeLearningOn: false,
      showConfidence: false,
      startAnnotation: false,
      autoSuggestStrength: 20,
      tradeoff: 0,

      text: 'Geoffrey Everest Hinton CC FRS FRSC (born 6 December 1947) is a British-Canadian cognitive psychologist and computer scientist, most noted for his work on artificial neural networks. Since 2013, he has divided his time working for Google (Google Brain) and the University of Toronto. \n\nIn 2017, he co-founded and became the Chief Scientific Advisor of the Vector Institute in Toronto. With David Rumelhart and Ronald J. Williams, Hinton was co-author of a highly cited paper published in 1986 that popularized the backpropagation algorithm for training multi-layer neural networks, although they were not the first to propose the approach. Hinton is viewed as a leading figure in the deep learning community. The dramatic image-recognition milestone of the AlexNet designed in collaboration with his students Alex Krizhevsky and Ilya Sutskever for the ImageNet challenge 2012 was a breakthrough in the field of computer vision. \n\nHinton received the 2018 Turing Award, together with Yoshua Bengio and Yann LeCun, for their work on deep learning. They are sometimes referred to as the "Godfathers of AI" and "Godfathers of Deep Learning", and have continued to give public talks together.',
      result: [],

      curEnt: "ORG",
      startIndex: null,
      dataIndex: null,

      historyResult: [],
      historyIndex: [],
      historyCurInd: -1,
    };
  },
  getters,
  mutations,
  actions,
};
