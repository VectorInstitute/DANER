const routes = [
  // Dashboard
  {
    path: "/",
    redirect: "/service",
  },

  // Machine Learning Model Detail
  {
    path: "/model",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/model/ModelList.vue"),
      },
    ],
  },
  {
    path: "/model/:id",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/model/ModelDetail.vue"),
      },
    ],
  },

  // Dataset Detail
  {
    path: "/data",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/data/DataList"),
      },
    ],
  },
  {
    path: "/data/:id",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/data/DataDetail.vue"),
      },
    ],
  },

  // NLP Service
  {
    path: "/service",
    redirect: "/service/ner",
  },

  {
    path: "/service/ner",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/service/NerService.vue"),
      },
    ],
  },

  // Data Annotation Page
  {
    path: "/annotation",
    redirect: "/annotation/ner",
  },

  {
    path: "/annotation/ner",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/annotation/NerAnnotation.vue"),
      },
    ],
  },

  // User

  {
    path: "/user/login",
    component: () => import("layouts/UserLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/user/Login.vue"),
      },
    ],
  },

  {
    path: "/user/register",
    component: () => import("layouts/UserLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/user/Register.vue"),
      },
    ],
  },

  // Exception
  {
    path: "/exception/403",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/exception/403"),
      },
    ],
  },
  {
    path: "/exception/404",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/exception/404"),
      },
    ],
  },
  {
    path: "/exception/500",
    component: () => import("layouts/MainLayout.vue"),
    children: [
      {
        path: "",
        component: () => import("pages/exception/500"),
      },
    ],
  },

  // Always leave this as last one,
  // but you can also remove it
  {
    path: "/:catchAll(.*)*",
    component: () => import("pages/exception/Error404.vue"),
  },
];

export default routes;
