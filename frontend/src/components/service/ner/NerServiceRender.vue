<template>
  <q-card v-if="nerSpans.length > 0">
    <q-card-section>
      <span v-for="(content, index) in contentList" :key="index">
        <p v-if="content.text === '<br>'"></p>
        <span v-else-if="content.color !== ''">
          <q-banner
            dense
            :style="{ background: getPaletteColor(content.color) }"
          >
            {{ content["text"] }}
            <span class="label-entity"> {{ content["label"] }} </span>
          </q-banner>
        </span>
        <span v-else>{{ content["text"] }}</span>
      </span>
    </q-card-section>
  </q-card>
</template>

<script>
import { computed } from "vue";
import { colors } from "quasar";

export default {
  props: ["nerText", "nerSpans", "ents", "entStyle"],

  setup(props) {
    const { getPaletteColor } = colors;
    let tmpColor = null;

    const contentList = computed(() => {
      let childNodes = [];
      let offset = 0;

      if (props.nerSpans.length > 0) {
        props.nerSpans.forEach(({ _, start, end, label }) => {
          const entity = props.nerText.slice(start, end);
          const fragments = props.nerText.slice(offset, start).split("\n");

          fragments.forEach((fragment, i) => {
            childNodes.push({ text: fragment, label: "", color: "" });
            if (fragments.length > 1 && i != fragments.length - 1)
              childNodes.push({ text: "<br>", label: "", color: "" });
          });

          if (props.ents.includes(label)) {
            tmpColor = props.entStyle.find(({ val }) => val === label).color;
            childNodes.push({ text: entity, label: label, color: tmpColor });
          } else {
            childNodes.push({ text: entity, label: "", color: "" });
          }
          offset = end;
        });
        childNodes.push({
          text: props.nerText.slice(offset, props.nerText.length),
          label: "",
          color: "",
        });
      }
      return childNodes;
    });
    return { contentList, getPaletteColor };
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
  padding: 0.35em 0.35em;
  border-radius: 0.35em;
  display: inline-block;
  font-weight: bold;
  margin: 0.1em;
}

</style>
