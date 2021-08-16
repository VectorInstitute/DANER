<template>
  <div>
    <q-layout view="hHh LpR fff">
      <q-header class="bg-white text-primary shadow-1">
        <q-toolbar>
          <q-toolbar-title> Sika Code Design</q-toolbar-title>
          <div>Sika v1.0.0</div>
        </q-toolbar>
      </q-header>
      <q-footer class="bg-white text-blue-grey-4">
        <div class="q-my-lg">
          <div class="text-center q-mb-sm">
            <span class="inline-block q-mr-sm">Sika Design Pro</span>
            <span class="inline-block q-ml-sm">Sika Design</span>
          </div>
        </div>
      </q-footer>
      <q-page-container>
        <div class="q-pa-sm q-mt-lg row justify-center">
          <div class="col text-center" style="width: 100%; max-width: 400px">
            <div class="q-mb-xl">
              <h4 class="q-ma-sm">
                <q-avatar class="q-mr-md">
                  <img alt="Sika logo" src="imgs/logo/sika-logo.png" />
                </q-avatar>
                <strong>Sika Design</strong>
              </h4>
            </div>
            <router-view />
          </div>
        </div>
      </q-page-container>
    </q-layout>
  </div>
</template>

<script>
export default {
  name: "UserLayout",
};
</script>
