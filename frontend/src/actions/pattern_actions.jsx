import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { CSS_COLOR_NAMES } from "../assets/color_assets";

import { base_url } from "../assets/base_url";
let controller = new AbortController();



export const explainPattern = createAsyncThunk(
    "workspace/explainpattern",
    async (request, { getState }) => {
      const state = getState();
      const { pattern } = request;
  
      var url = new URL(`${base_url}/explain/${pattern}`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "GET",
      }).then((response) => response.json());
  
      return data;
    }
  );
  
  export const deletePattern = createAsyncThunk(
    "workspace/deletePattern",
    async (request, { getState }) => {
      const { theme, pattern } = request;
      var url = new URL(`${base_url}/delete_pattern`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({ theme: theme, pattern: pattern }),
      }).then((response) => response.json());
  
      return data;
    }
  );
  
  export const pinPattern = createAsyncThunk(
    "workspace/pinPattern",
    async (request, { getState }) => {
      const { theme, pattern } = request;
      var url = new URL(`${base_url}/pin_pattern`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({ theme: theme, pattern: pattern }),
      }).then((response) => response.json());
  
      return data;
    }
  );


export const fetchRelatedExample = createAsyncThunk(
    "workspace/related_examples",
    async (request, { getState }) => {
      const state = getState();
  
      const { id } = request;
  
      var url = new URL(`${base_url}/related_examples/${id}`);
  
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

  export const fetchPatterns = createAsyncThunk(
    "workspace/patterns",
    async (request, { getState }) => {
      const signal = controller.signal;
  
      var url = new URL(`${base_url}/patterns`);
  
      const data = await fetch(url, {
        signal: signal,
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

  export const fetchCombinedPatterns = createAsyncThunk(
    "workspace/combinedpatterns",
    async (request, { getState }) => {
      const signal = controller.signal;
  
      var url = new URL(`${base_url}/annotations`);
  
      const data = await fetch(url, {
        signal: signal,
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

  export const deleteSoftmatch = createAsyncThunk(
    "workspace/delete_softmatch_globally",
    async (request, { getState }) => {
      const { pivot_word, similar_word } = request;
  
      var url = new URL(
        `${base_url}/delete_softmatch_globally/${pivot_word}/${similar_word}`
      );
  
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
  