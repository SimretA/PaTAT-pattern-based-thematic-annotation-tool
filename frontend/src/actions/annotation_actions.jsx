import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { CSS_COLOR_NAMES } from "../assets/color_assets";

import { base_url } from "../assets/base_url";


export const clearAnnotation = createAsyncThunk(
    "workspace/clear",
    async (request, { getState }) => {
      const state = getState();
  
      var url = new URL(`${base_url}/clear`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
      }).then((response) => response.json());
  
      return data;
    }
  );

  
  export const multiLabelData = createAsyncThunk(
    "workspace/multiLabelData",
    async (request, { getState }) => {
      const { elementId, label, positive } = request;
  
      if (positive == 0) {
        var requestBody = { elementId: elementId, theme: label, positive: 0 };
      } else {
        var requestBody = { elementId: elementId, theme: label, positive: 1 };
      }
  
      const data = await fetch(`${base_url}/label`, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify(requestBody),
      }).then((response) => response.json());
  
      return data;
    }
  );

  export const deleteLabelData = createAsyncThunk(
    "workspace/deleteLabelData",
    async (request, { getState }) => {
      const { elementId, label } = request;
      var url = new URL(`${base_url}/delete_label`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({ elementId: elementId, theme: label }),
      }).then((response) => response.json());
  
      return data;
    }
  );

  export const labelPhrase = createAsyncThunk(
    "workspace/phrase",
    async (request, { getState }) => {
      const { phrase, label, id, positive } = request;
  
      var url = new URL(`${base_url}/phrase`);
  
      let data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({
          phrase: phrase,
          theme: label,
          elementId: id,
          positive: positive,
        }),
      }).then((response) => response.json());
  
      data["id"] = id;
      data["phrase"] = phrase;
      data["label"] = label;
      data["positive"] = positive;
      return data;
    }
  );


export const toggleBinaryMode = createAsyncThunk(
    "workspace/toggleBinaryMode",
    async (request, { getState }) => {
      const { binary_mode } = request;
      const binary_mode_value = binary_mode ? 0 : 1;
  
      var url = new URL(`${base_url}/toggle_binary_mode/${binary_mode_value}`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "GET",
      }).then((response) => response.json());
  
      return data;
    }
  );


export const groupAnnotationsRemote = createAsyncThunk(
    "workspace/group_annotations",
    async (request, { getState }) => {
      const state = getState();
  
      const { ids, label } = request;
  
      var url = new URL(`${base_url}/bulk_label`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify(request),
      }).then((response) => response.json());
  
      return data;
    }
  );
  