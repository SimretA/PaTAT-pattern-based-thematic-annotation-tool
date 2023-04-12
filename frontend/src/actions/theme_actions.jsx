import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { CSS_COLOR_NAMES } from "../assets/color_assets";

import { base_url } from "../assets/base_url";



export const fetchThemes = createAsyncThunk(
    "workspace/themes",
    async (request, { getState }) => {
      var url = new URL(`${base_url}/themes`);
  
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
  
  export const fetchSelectedTheme = createAsyncThunk(
    "workspace/selected_theme",
    async (request, { getState }) => {
      var url = new URL(`${base_url}/selected_theme`);
  
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

  
  export const mergeThemes = createAsyncThunk(
    "workspace/merge_themes",
    async (request, { getState }) => {
  
      var url = new URL(`${base_url}/merge_themes`);
  
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
  
  export const splitTheme = createAsyncThunk(
    "workspace/split_themes",
    async (request, { getState }) => {
      const { theme, group1, group2 } = request;
  
      var url = new URL(`${base_url}/split_theme`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({ theme: theme, group1, group2 }),
      }).then((response) => response.json());
  
      return data;
    }
  );
  
  export const splitThemeByPattern = createAsyncThunk(
    "workspace/split_theme_by_pattern",
    async (request, { getState }) => {
      const { theme, patterns, new_theme_name } = request;
  
      var url = new URL(`${base_url}/split_theme_by_pattern`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({ theme: theme, patterns, new_theme_name }),
      }).then((response) => response.json());
  
      return data;
    }
  );
  
  export const renameThemeRemote = createAsyncThunk(
    "workspace/rename_theme",
    async (request, { getState }) => {
      const { theme, new_name } = request;
  
      var url = new URL(`${base_url}/rename_theme`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({
          theme: theme.toLowerCase(),
          new_name: new_name.toLowerCase(),
        }),
      }).then((response) => response.json());
  
      return data;
    }
  );


export const deleteTheme = createAsyncThunk(
    "workspace/deleteTheme",
    async (request, { getState }) => {
      const { theme } = request;
      var url = new URL(`${base_url}/delete_theme`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({ theme: theme }),
      }).then((response) => response.json());
  
      return data;
    }
  );
  

  export const addThemeRemote = createAsyncThunk(
    "workspace/addTheme",
    async (request, { getState }) => {
      const { theme } = request;
  
      var url = new URL(`${base_url}/add_theme`);
  
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
  
  export const setTheme = createAsyncThunk(
    "workspace/setTheme",
    async (request, { getState }) => {
      const { theme } = request;
  
      var url = new URL(`${base_url}/set_theme`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({ theme: theme.toLowerCase() }),
      }).then((response) => response.json());
  
      return data;
    }
  );
  