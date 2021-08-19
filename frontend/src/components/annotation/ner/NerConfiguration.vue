<template>
  <q-card>
    <q-card-section class="bg-primary text-white q-pa-sm" align="center">
      <div class="text-h4">Annotation Configuration</div>
    </q-card-section>

    <q-card-section>
      <div class="row" style="">
        <div class="col">
          <div class="q-gutter-md row">
            <q-select
              filled
              v-model="dataset"
              :options="ALL_DATASETS"
              label="Datasets"
              style="width: 150px"
            />
            <q-select
              filled
              v-model="model"
              :options="ALL_MODELS"
              label="Model"
              style="width: 150px"
            />
          </div>
          <br />
          <div class="q-gutter-sm">
            <q-checkbox
              v-model="ents"
              keep-color
              v-for="(ent, index) in ALL_ENTS"
              :key="index"
              :val="ent.val"
              :label="ent.label"
              :color="ent.color"
            />
          </div>
          <q-btn color="primary" label="Select All" @click="onSelectALL" />
          <q-btn
            color="secondary"
            label="Default"
            @click="onSelectDefault"
            class="q-ma-md"
          />
          <q-btn
            color="primary"
            label="Reset"
            flat
            @click="onResetLabel"
            class="q-ma-md"
          />
        </div>

        <q-separator vertical />

        <div class="col q-ma-sm">
          <div>
            <q-toggle label="Auto Suggestion" v-model="showConfidence" />
            <q-item v-if="showConfidence">
              <q-item-section side> Confidence Level: 0 </q-item-section>
              <q-item-section>
                <q-slider
                  v-model="autoSuggestStrength"
                  :min="0"
                  :max="100"
                  label
                />
              </q-item-section>
              <q-item-section side> 100 </q-item-section>
            </q-item>
          </div>

          <div>
            <div class="q-gutter-md row">
              <q-toggle label="Active Learning" v-model="activeLearningOn" />
              <q-select
                v-if="activeLearningOn"
                filled
                v-model="al"
                :options="ALL_ALS"
                label="AL Algorithm"
                style="width: 150px"
              />
            </div>

            <!-- <div>
                <q-item>
                  <q-item-section side>
                    Uncertainty and Diversity Tradeoff: Uncertainty
                  </q-item-section>
                  <q-item-section>
                    <q-slider v-model="tradeoff" :min="0" :max="100" label />
                  </q-item-section>
                  <q-item-section side> Diversity </q-item-section>
                </q-item>
              </div> -->
          </div>

          <div class="q-gutter-md">
            <div class="q-gutter-md row items-start">
              <q-btn
                color="primary"
                label="start"
                @click="onStartAnnotation"
                class="q-ma-md"
              />
            </div>
          </div>

          <!-- The following section only appears in demo to show the stage of active learning-->
          <div v-if="startAnnotation" class="q-mt-md">
            <q-separator horizontal />
            <div class="q-mt-md">
              <q-toggle
                label="Early Phase (Demo Only)"
                v-model="earlyPhaseOn"
              />
            </div>
          </div>
        </div>
      </div>
    </q-card-section>
  </q-card>
</template>

<script>
import { useQuasar } from "quasar";
import { ref, computed } from "vue";
import { useStore } from "vuex";

export default {
  setup() {
    const $q = useQuasar();
    const store = useStore();

    const earlyPhaseOn = computed({
      get: () => store.state.annotationNer.earlyPhaseOn,
      set: (val) => {
        store.commit("annotationNer/updateEarlyPhaseOn", val);
      },
    });
    const activeLearningOn = computed({
      get: () => store.state.annotationNer.activeLearningOn,
      set: (val) => {
        store.commit("annotationNer/updateActiveLearningOn", val);
      },
    });

    const showConfidence = computed({
      get: () => store.state.annotationNer.showConfidence,
      set: (val) => {
        store.commit("annotationNer/updateShowConfidence", val);
      },
    });

    const autoSuggestStrength = computed({
      get: () => store.state.annotationNer.autoSuggestStrength,
      set: (val) => {
        store.commit("annotationNer/updateAutoSuggestStrength", val);
      },
    });
    const tradeoff = computed({
      get: () => store.state.annotationNer.tradeoff,
      set: (val) => {
        store.commit("annotationNer/updateTradeoff", val);
      },
    });

    const startAnnotation = computed(
      () => store.state.annotationNer.startAnnotation
    );
    const ALL_MODELS = store.state.annotationNer.ALL_MODELS;
    const ALL_DATASETS = store.state.annotationNer.ALL_DATASETS;
    const ALL_ALS = store.state.annotationNer.ALL_ALS;
    const ALL_ENTS = store.state.annotationNer.ALL_ENTS;

    const textRef = ref(null);
    const text = computed({
      get: () => store.state.annotationNer.text,
      set: (val) => {
        store.commit("annotationNer/updateText", val);
      },
    });

    const ents = computed({
      get: () => store.state.annotationNer.ents,
      set: (val) => {
        store.commit("annotationNer/updateEnts", val);
      },
    });

    const model = computed({
      get: () => store.state.annotationNer.model,
      set: (val) => {
        store.commit("annotationNer/updateModel", val);
      },
    });

    const dataset = computed({
      get: () => store.state.annotationNer.dataset,
      set: (val) => {
        store.commit("annotationNer/updateDataset", val);
      },
    });

    const al = computed({
      get: () => store.state.annotationNer.al,
      set: (val) => {
        store.commit("annotationNer/updateAL", val);
      },
    });

    const result = computed(() => store.state.annotationNer.result);

    const textRules = [
      (val) => (val && val.length > 0) || "Please type something to analyze!",
      (val) => (val && val.length < 1500) || "Text too long",
    ];

    function onSelectALL() {
      ents.value = store.getters["annotationNer/getAllEnts"];
    }
    function onSelectDefault() {
      ents.value = store.getters["annotationNer/getDefaultEnts"];
    }
    function onResetLabel() {
      ents.value = [];
    }
    function onStartAnnotation() {
      store.commit("annotationNer/updateStartAnnotation", true);
      store.dispatch("annotationNer/getDataset");
      // Todo: Upload the configuration to backend
    }

    function onSubmit() {
      textRef.value.validate();

      if (textRef.value.hasError) {
        $q.notify({
          color: "negative",
          message: "No content to analyze",
        });
      } else {
        $q.notify({
          icon: "done",
          color: "positive",
          message: "Send to server for analyze",
        });
      }
      store.dispatch("annotationNer/analyzeText", { textInput: text.value });
    }

    function onReset() {
      text.value = null;
      textRef.value.resetValidation();
    }

    return {
      earlyPhaseOn,
      activeLearningOn,
      showConfidence,
      autoSuggestStrength,
      tradeoff,
      text,
      textRef,
      textRules,
      result,
      ents,
      model,
      dataset,
      al,
      startAnnotation,
      ALL_ENTS,
      ALL_MODELS,
      ALL_DATASETS,
      ALL_ALS,
      onSelectALL,
      onSelectDefault,
      onResetLabel,
      onStartAnnotation,
      onSubmit,
      onReset,
    };
  },
};
</script>
