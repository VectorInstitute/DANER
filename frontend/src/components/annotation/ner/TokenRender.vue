<template>
  <span :id="id">
    <q-btn
      v-if="token.color !== '' && token.confidence >= conf"
      dense
      flat
      no-caps
      :style="{ background: getColor(token) }"
    >
      {{ token.token }}
      <span v-if="token.color !== ''" class="label-entity">
        {{ token.label }}
      </span>
    </q-btn>
    <q-banner v-else dense>
      {{ token.token }}
    </q-banner>
    <q-tooltip v-if="showConfidence" :offset="[0, 4]">{{
      token.confidence.toFixed(3)
    }}</q-tooltip>
  </span>
</template>

<script>
import { colors } from "quasar";
import { computed } from "vue";
import { useStore } from "vuex";

export default {
  props: ["token", "id"],
  setup() {
    const store = useStore();
    const { getPaletteColor } = colors;

    const showConfidence = computed({
      get: () => store.state.annotationNer.showConfidence,
    });

    const conf = computed({
      get: () => store.state.annotationNer.autoSuggestStrength / 100,
    });

    function getColor(token) {
      if (token.color !== "") {
        return getPaletteColor(token.color);
      } else {
        return "#fff";
      }
    }
    return {
      conf,
      showConfidence,
      getColor,
    };
  },
};
</script>

<style lang="scss" scoped>
.label-entity {
  -webkit-box-sizing: border-box;
  box-sizing: border-box;
  font-size: 0.7em;
  line-height: 1;
  padding: 0.35em 0.35em;
  border-radius: 0.35em;
  text-transform: uppercase;
  display: inline-block;
  vertical-align: middle;
  margin: 0 0 0.15rem 0.35rem;
  background: #fff;
  font-weight: bold;
}
.q-banner {
  -webkit-box-sizing: border-box;
  box-sizing: border-box;
  line-height: 1;
  padding: 0.2em;
  border-radius: 0.35em;
  display: inline-block;
  font-weight: bold;
  margin: 0.1em;
}
</style>
