import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { CSS_COLOR_NAMES } from "../assets/color_assets";

import { base_url } from "../assets/base_url";

export const fetchDataset = createAsyncThunk(
    "workspace/dataset",
    async (request, { getState }) => {
      var url = new URL(`${base_url}/dataset`);
  
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
  
  export const fetchUserlabeledData = createAsyncThunk(
    "workspace/labeled_data",
    async (request, { getState }) => {
      const { theme } = request;
      var url = new URL(`${base_url}/labeled_data`);
  
      const data = await fetch(url, {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
          credentials: "include",
          annotuser: window.localStorage.getItem("user").replaceAll('"', ""),
        },
        method: "POST",
        body: JSON.stringify({ theme: theme }),
      }).then((response) => response.json());
  
      return data;
    }
  );
  
  export const fetchGroupedDataset = createAsyncThunk(
    "workspace/groupedDataset",
    async (request, { getState }) => {
      const { selectedSetting } = request;
      if (selectedSetting == 0) {
        var url = new URL(`${base_url}/original_dataset_order`);
      } else if (selectedSetting == 1) {
        var url = new URL(`${base_url}/pattern_clusters`);
      } else if (selectedSetting == 2) {
        var url = new URL(`${base_url}/NN_classification`);
      } else if (selectedSetting == 3) {
        var url = new URL(`${base_url}/NN_cluster`);
      }
  
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
  

  export const reorderDataset = (dataset, setting, elements, groups) => {
  
    let reorderedGroups = [...groups];
    switch (`${setting}`) {
      case "0": //Do not Reorder
        return reorderedGroups;
        break;
      case "1":
        groups.forEach((elementIds, index) => {
          elementIds.sort(function (a, b) {
            return -elements[a].score + elements[b].score;
          });
  
          reorderedGroups[index] = elementIds;
        });
        return reorderedGroups;
  
        break;
      case "2":
        groups.forEach((elementIds, index) => {
          elementIds.sort(function (a, b) {
            return elements[a].score - elements[b].score;
          });
  
          reorderedGroups[index] = elementIds;
        });
        return reorderedGroups;
  
        break;
      case "3":
        groups.forEach((elementIds, index) => {
          elementIds.sort(function (a, b) {
            return (
              Math.abs(elements[a].score - 0.5) -
              Math.abs(elements[b].score - 0.5)
            );
          });
  
          reorderedGroups[index] = elementIds;
        });
        return reorderedGroups;
        break;
      case "4":
        // code block
  
        return reorderedGroups;
        break;
      default:
        dataset.sort(function (a, b) {
          return elements[a].score - elements[b].score;
        });
      // code block
    }
  
  };
  


export const createSession = createAsyncThunk(
  "workspace/create_session",
  async (request, { getState }) => {
    const state = getState();
    const { user } = request;

    var url = new URL(`${base_url}/create_session/${user}`);

    const data = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        credentials: "include",
        annotuser: user,
      },
      method: "GET",
    }).then((response) => response.json());

    return data;
  }
);

