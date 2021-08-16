<template>
  <q-layout view="hHh Lpr lff">
    <q-header bordered class="bg-primary text-white" height-hint="98">
      <q-toolbar>
        <q-btn
          flat
          dense
          round
          icon="menu"
          aria-label="Menu"
          @click="toggleLeftDrawer"
        />
        <q-toolbar-title> NERA </q-toolbar-title>
      </q-toolbar>
    </q-header>

    <q-drawer
      v-model="leftDrawerOpen"
      show-if-above
      :mini="miniState"
      @mouseover="miniState = false"
      @mouseout="miniState = true"
      mini-to-overlay
      :width="200"
      :breakpoint="767"
      bordered
      class="bg-grey-1"
    >
      <q-scroll-area class="fit">
        <q-list padding>
          <q-item
            v-for="nav in navs"
            :key="nav.label"
            :to="nav.to"
            exact
            clickable
            v-ripple
          >
            <q-item-section avatar>
              <q-icon :name="nav.icon" />
            </q-item-section>

            <q-item-section>
              <q-item-label>{{ nav.label }}</q-item-label>
            </q-item-section>
          </q-item>

          <q-separator dark />
        </q-list>
      </q-scroll-area>
    </q-drawer>

    <q-page-container>
      <!-- <q-page-sticky
        expand
        position="top"
        style="z-index: 2000"
        class="q-py-xs bg-grey-2"
      >
        <q-toolbar style="min-height: 40px">
          <q-tabs
            dense
            shrink
            inline-label
            indicator-color="transparent"
            active-bg-color="primary"
            active-color="white"
            v-model="currentPath"
          >
            <div
              v-for="(tab, index) in tabs"
              :key="index"
              class="q-mr-sm items-center"
            >
              <q-tab
                :ripple="false"
                :label="tab.name"
                :name="tab.to"
                @click="routeTo(tab.to)"
                :class="{
                  'bg-white': !tab.active,
                  'text-grey-7': !tab.active,
                }"
                exact
                style="padding: 0 8px; min-height: 24px; border-radius: 4px"
              >
                <q-icon
                  v-if="tab.to !== '/dashboard/analysis'"
                  size="18px"
                  name="close"
                  @click.stop="removeTab(index)"
                ></q-icon>
              </q-tab>
            </div>
          </q-tabs>
        </q-toolbar>
      </q-page-sticky> -->
      <transition mode="out-in">
        <router-view />
      </transition>
    </q-page-container>

    <q-footer>
      <q-tabs>
        <q-route-tab
          v-for="nav in navs"
          :key="nav.label"
          :to="nav.to"
          :icon="nav.icon"
          :label="nav.label"
        />
      </q-tabs>
    </q-footer>
  </q-layout>
</template>

<script>
const navs = [
  // {
  //   label: "Data",
  //   icon: "perm_media",
  //   to: "/data",
  // },
  // {
  //   label: "Model",
  //   icon: "model_training",
  //   to: "/model",
  // },
  {
    label: "Service",
    icon: "launch",
    to: "/service",
  },
  {
    label: "Annotation",
    icon: "touch_app",
    to: "/annotation",
  },
];

import { defineComponent, ref } from "vue";

export default defineComponent({
  name: "MainLayout",

  setup() {
    const leftDrawerOpen = ref(false);
    const miniState = ref(true);
    const currentPath = ref(null);
    const tabs = [];

    return {
      navs,
      tabs,
      leftDrawerOpen,
      miniState,
      currentPath,
      toggleLeftDrawer() {
        leftDrawerOpen.value = !leftDrawerOpen.value;
      },
      // buildTabRoute() {
      //   this.currentPath = LAYOUT_DATA.addTab(this.$route.path).to;
      // },
      // routeTo(to) {
      //   this.$router.push(to).catch((err) => {
      //     console.log(err);
      //   });
      // },
      // removeTab(index) {
      //   this.tabs.splice(index, 1);
      //   let to = "/";
      //   if (this.tabs.length > 0) {
      //     to = this.tabs[this.tabs.length - 1].to;
      //   }
      //   this.$router.push(to).catch((err) => {
      //     console.log(err);
      //   });
      // },
    };
  },
});
</script>

<style lang="scss">
@media screen and (min-width: 768px) {
  .q-footer {
    display: none;
  }
}
// .q-drawer {
//   .q-router-link--exact-active {
//     color: white !important;
//   }
// }
</style>
