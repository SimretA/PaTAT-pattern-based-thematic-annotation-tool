import { configureStore } from "@reduxjs/toolkit";

import WorkspaceReducer from "../actions/Dataslice";

export default configureStore({
  reducer: {
    workspace: WorkspaceReducer,
  },
});
