import { createSlice, createAsyncThunk } from "@reduxjs/toolkit";
import { CSS_COLOR_NAMES } from "../assets/color_assets";
import {createSession, fetchDataset, fetchGroupedDataset, fetchUserlabeledData, reorderDataset } from "./dataset_actions";

import { multiLabelData, clearAnnotation, deleteLabelData, labelPhrase, toggleBinaryMode, groupAnnotationsRemote } from "./annotation_actions";

import { setTheme, addThemeRemote ,deleteTheme  ,renameThemeRemote, splitThemeByPattern,splitTheme, mergeThemes, fetchSelectedTheme, fetchThemes } from "./theme_actions";


import { explainPattern, deletePattern, pinPattern, fetchRelatedExample, fetchPatterns, fetchCombinedPatterns, deleteSoftmatch } from "./pattern_actions";

import { base_url } from "../assets/base_url";


let controller = new AbortController();
const settingsEnum = Object.freeze({
  0: "Default",
  1: "Positives",
  2: "Negatives",
  3: "Unsure",
});

const groupingSettings = Object.freeze({
  0: "None",
  1: "Patterns",
  2: "Model predictions",
  3: "Similarity",
});

let patterns_cache = {};
let combinedPatterns_cache = {};

let explanations_cache = {};
let selectedPatterns_cache = {};
let modelannotationCount_cache = {};

const initialState = {
  workspace: "fairytale-bias-val-split",
  dataset: [],
  groups: [],
  groupNames: [],
  backupGroups: [],

  elements: {},
  userLabel: {},
  modelLabel: {},
  rules: null,
  combinedRules: null,
  curCategory: null,
  scores: null,

  patterns: [],
  combinedPatterns: {},
  explanation: {},
  modelAnnotationCount: 0,

  loading: true,
  loadingCombinedPatterns: false,
  loadingPatterns: false,
  orderSetting: settingsEnum,
  groupingSettings: groupingSettings,
  selectedSetting: 0,
  selectedGroupSetting: 0,
  annotationPerRetrain: 10,
  userAnnotationCount: 0,
  userAnnotationTracker: 0,
  totalDataset: 0,
  selectedPatterns: {},
  themes: [],
  relatedExamples: [],
  selectedTheme: null,
  element_to_label: {},
  negative_element_to_label: {},
  color_code: {},
  not_color: "#fc0b22",
  binary_mode: false,
  refresh: false,

  cacheHit: null,
  color_schema: CSS_COLOR_NAMES,
};


const get_user_annotation_count = (
  element_to_label,
  negative_element_to_label,
  theme
) => {
  let userAnnotationCount = 0;
  element_to_label &&
    Object.keys(element_to_label).forEach((id) => {
      const label_list = element_to_label[id];
      const found = label_list && label_list.find((el) => el == theme);
      if (found) {
        userAnnotationCount += 1;
      }
    });

  negative_element_to_label &&
    Object.keys(negative_element_to_label).forEach((id) => {
      const label_list = negative_element_to_label[id];
      const found = label_list && label_list.find((el) => el == theme);
      if (found) {
        userAnnotationCount += 1;
      }
    });

  return userAnnotationCount;
};


const DataSlice = createSlice({
  name: "workspace",
  initialState,
  reducers: {
    addTheme(state, action) {
      let themes = JSON.parse(JSON.stringify(state.themes));
      if (themes && themes.find((el) => el == action.payload.theme)) {
        return {
          ...state,
        };
      }

      let color_code = JSON.parse(JSON.stringify(state.color_code));

      const newTheme = action.payload.theme;
      const index = action.payload.index;

      //TODO input validation

      let color_schema = JSON.parse(JSON.stringify(state.color_schema));
      themes = [...themes, newTheme];
      color_code[`${newTheme}`] = color_schema[index];

      return {
        ...state,
        themes,
        color_code,
      };
    },

    updateElementLabel(state, action) {
      let refresh = false;

      let selectedTheme = JSON.parse(JSON.stringify(state.selectedTheme));

      let userAnnotationTracker = JSON.parse(
        JSON.stringify(state.userAnnotationTracker)
      );
      let negative_element_to_label = JSON.parse(
        JSON.stringify(state.negative_element_to_label)
      );

      if (action.payload.event == "REMOVE") {
        let element_to_label = JSON.parse(
          JSON.stringify(state.element_to_label)
        );

        if (element_to_label[action.payload.elementId]) {
          let userAnnotationCount = JSON.parse(
            JSON.stringify(state.userAnnotationCount)
          );
          element_to_label[action.payload.elementId] = element_to_label[
            action.payload.elementId
          ].filter((word) => word != action.payload.label);
          refresh = true;
          return {
            ...state,
            element_to_label,
            userAnnotationCount: userAnnotationCount - 1,
            refresh,
          };
        }
      } else if (action.payload.event == "ADD") {
        //check if label is already in negative batch
        const found =
          negative_element_to_label[action.payload.elementId] &&
          negative_element_to_label[action.payload.elementId].find(
            (el) => el == action.payload.label
          );
        if (found) {
          negative_element_to_label[action.payload.elementId] =
            negative_element_to_label[action.payload.elementId].filter(
              (el) => el != action.payload.label
            );
        }

        let element_to_label = JSON.parse(
          JSON.stringify(state.element_to_label)
        );

        if (
          !element_to_label[action.payload.elementId] ||
          element_to_label[action.payload.elementId].indexOf(
            action.payload.label
          ) == -1
        ) {
          userAnnotationTracker += 1;
          refresh = true;
        }

        if (element_to_label[action.payload.elementId]) {
          element_to_label[action.payload.elementId] = [
            ...element_to_label[action.payload.elementId],
            action.payload.label,
          ];
        } else {
          element_to_label[action.payload.elementId] = [action.payload.label];
        }

        let userAnnotationCount = get_user_annotation_count(
          element_to_label,
          negative_element_to_label,
          selectedTheme
        );

        return {
          ...state,
          element_to_label,
          negative_element_to_label,
          userAnnotationCount: userAnnotationCount,
          userAnnotationTracker,
          refresh: refresh,
        };
      }
    },
    updateNegativeElementLabel(state, action) {
      let refresh = false;

      const { elementId, theme, label } = action.payload;
      let element_to_label = JSON.parse(JSON.stringify(state.element_to_label));
      let negative_element_to_label = JSON.parse(
        JSON.stringify(state.negative_element_to_label)
      );

      let selectedTheme = JSON.parse(JSON.stringify(state.selectedTheme));

      let userAnnotationTracker = JSON.parse(
        JSON.stringify(state.userAnnotationTracker)
      );

      userAnnotationTracker += 1;
      refresh = true;

      //check if already labeled in the opposite collection
      //add to list of given label
      if (label == 1) {
        //check in negative element to label collection
        negative_element_to_label[elementId] =
          negative_element_to_label[elementId] &&
          negative_element_to_label[elementId].filter(
            (label) => label != theme
          );
        if (element_to_label[action.payload.elementId]) {
          element_to_label[action.payload.elementId] = [
            ...element_to_label[action.payload.elementId],
            action.payload.theme,
          ];
        } else {
          element_to_label[action.payload.elementId] = [action.payload.theme];
        }
      } else if (label == 0) {
        //check in positive element to label collection
        element_to_label[elementId] =
          element_to_label[elementId] &&
          element_to_label[elementId].filter((label) => label != theme);

        if (negative_element_to_label[action.payload.elementId]) {
          negative_element_to_label[action.payload.elementId] = [
            ...new Set([
              ...negative_element_to_label[action.payload.elementId],
              action.payload.theme,
            ]),
          ];
        } else {
          negative_element_to_label[action.payload.elementId] = [
            action.payload.theme,
          ];
        }
      }
      let userAnnotationCount = get_user_annotation_count(
        element_to_label,
        negative_element_to_label,
        selectedTheme
      );

      return {
        ...state,
        negative_element_to_label,
        element_to_label,
        userAnnotationCount,
        userAnnotationTracker,
        refresh: refresh,
      };
    },
    updatePatExp(state, action) {
      let patternExp = JSON.parse(JSON.stringify(state.patternExp));
      let { pattern, soft_match } = action.payload;

      patternExp[pattern][1] = patternExp[pattern][1].filter(
        (match) => match != soft_match
      );

      return {
        ...state,
        patternExp,
      };
    },
    updateBinaryMode(state, action) {
      let binary_mode = JSON.parse(JSON.stringify(state.binary_mode));

      return {
        ...state,
        binary_mode: !binary_mode,
      };
    },
    updatePatterns(state, action) {
      const { pattern, status } = action.payload;
      let patterns = JSON.parse(JSON.stringify(state.patterns));

      let combinedPatterns = JSON.parse(JSON.stringify(state.combinedPatterns));
      try {
        patterns[pattern]["status"] = status;
      } catch {
        console.log("This shouldnt happen ", patterns);
      }
      if (status == 0) {

        combinedPatterns["patterns"] = combinedPatterns["patterns"].filter(
          function (el) {
            return el["pattern"] != pattern;
          }
        );
      } else {

        let patt = patterns[pattern];
        patt["weight"] = "-";
        combinedPatterns["patterns"].push(patt);
      }

      return {
        ...state,
        patterns: patterns,
        combinedPatterns: combinedPatterns,
      };
    },
    changeSetting(state, action) {
      let dataset = JSON.parse(JSON.stringify(state.dataset));
      let elements = JSON.parse(JSON.stringify(state.elements));

      const backupGroups = JSON.parse(JSON.stringify(state.backupGroups));

      let groups = JSON.parse(JSON.stringify(state.groups));
      let reorderedGroups = JSON.parse(JSON.stringify(state.groups));
      let selectedTheme = JSON.parse(JSON.stringify(state.selectedTheme));
      const selectedGroupSetting = JSON.parse(
        JSON.stringify(state.selectedGroupSetting)
      );

      const combinedPatterns = JSON.parse(
        JSON.stringify(state.combinedPatterns)
      );
      if (Object.keys(combinedPatterns).length > 0) {
        reorderedGroups = reorderDataset(
          dataset,
          action.payload.selectedSetting,
          elements,
          groups
        );
      }
      return {
        ...state,
        selectedSetting: action.payload.selectedSetting,
        dataset: dataset,
        groups:
          action.payload.selectedSetting != 0 ? reorderedGroups : backupGroups,
      };
    },

    changeGroupingSetting(state, action) {
      return {
        ...state,
        selectedGroupSetting: action.payload.selectedSetting,
        selectedSetting: 0,
      };
    },

    clearAnnotations(state, action) {
      let elements = JSON.parse(JSON.stringify(state.elements));
      Object.keys(elements).map((elementId) => {
        elements[elementId]["score"] = null;
      });

      return {
        ...state,
        elements: elements,
      };
    },
    renameThemeLocal(state, action) {
      const { theme, new_name } = action.payload;
      let element_to_label = JSON.parse(JSON.stringify(state.element_to_label));

      let negative_element_to_label = JSON.parse(
        JSON.stringify(state.negative_element_to_label)
      );

      let all_themes = JSON.parse(JSON.stringify(state.themes));

      let color_code = JSON.parse(JSON.stringify(state.color_code));
      color_code[new_name] = color_code[theme];
      let selectedTheme = JSON.parse(JSON.stringify(state.selectedTheme));
      Object.keys(element_to_label).map((elementId) => {
        let index = element_to_label[elementId].indexOf(theme);
        if (index != -1) {
          element_to_label[elementId][index] = new_name;
        }
      });

      Object.keys(negative_element_to_label).map((elementId) => {
        let index = negative_element_to_label[elementId].indexOf(theme);
        if (index != -1) {
          negative_element_to_label[elementId][index] = new_name;
        }
      });

      let index = all_themes.indexOf(theme);
      all_themes[index] = new_name;
      if (selectedTheme == theme) {
        selectedTheme = new_name;
      }

      return {
        ...state,
        element_to_label: element_to_label,
        themes: all_themes,
        selectedTheme: selectedTheme,
        color_code: color_code,
        negative_element_to_label: negative_element_to_label,
      };
    },

    groupAnnotations(state, action) {
      const { ids, label, positive } = action.payload;

      let userAnnotationCount = JSON.parse(
        JSON.stringify(state.userAnnotationCount)
      );
      let userAnnotationTracker = JSON.parse(
        JSON.stringify(state.userAnnotationTracker)
      );

      let element_to_label = JSON.parse(JSON.stringify(state.element_to_label));

      let negative_element_to_label = JSON.parse(
        JSON.stringify(state.negative_element_to_label)
      );

      if (positive == 1) {
        ids.forEach((elementId) => {
          userAnnotationCount += 1;
          userAnnotationTracker += 1;
          if (element_to_label[elementId]) {
            element_to_label[elementId] = [
              ...element_to_label[elementId],
              label,
            ];
          } else {
            element_to_label[elementId] = [label];
          }
        });
      } else {
        ids.forEach((elementId) => {
          userAnnotationCount += 1;
          if (negative_element_to_label[elementId]) {
            negative_element_to_label[elementId] = [
              ...negative_element_to_label[elementId],
              label,
            ];
          } else {
            negative_element_to_label[elementId] = [label];
          }
        });
      }

      return {
        ...state,
        element_to_label: element_to_label,
        negative_element_to_label: negative_element_to_label,
        userAnnotationCount: userAnnotationCount,
        userAnnotationTracker: userAnnotationTracker,
      };
    },
    clearHighlight(state, action) {
      const { key, label, start, end, patterns, elementId } = action.payload;
      let explanation = JSON.parse(JSON.stringify(state.explanation));
      patterns &&
        patterns.forEach((pattern) => {
          let pattern_exp = explanation[pattern][elementId] || [];

          if (pattern_exp != "" || true) {
            pattern_exp.forEach((exp) => {
              pattern_exp = pattern_exp.filter(
                (explanation) =>
                  explanation[1] != start && explanation[2] != end
              );

              explanation[pattern][elementId] = pattern_exp;
            });
          }
        });

      return {
        ...state,
        explanation: explanation,
      };
    },

    getCache(state, action) {
      const selectedTheme = JSON.parse(JSON.stringify(state.selectedTheme));
      if (
        patterns_cache[selectedTheme] == null ||
        patterns_cache[selectedTheme] == undefined
      ) {

        return {
          ...state,
          cacheHit: false,
        };
      }
     

      return {
        ...state,
        patterns: patterns_cache[selectedTheme] || [],
        combinedPatterns: combinedPatterns_cache[selectedTheme] || {},
        explanation: explanations_cache[selectedTheme] || {},
        selectedPatterns: selectedPatterns_cache[selectedTheme] || {},
        modelAnnotationCount: modelannotationCount_cache[selectedTheme] || 0,
        loadingCombinedPatterns: false,
        loadingPatterns: false,
        refresh: false,
        cacheHit: true,
      };
    },
    abortApiCall(state, action) {
      if (controller) {
        console.log(
          "LOGG Aborting becase theme is ",
          JSON.parse(JSON.stringify(state.selectedTheme))
        );
        controller.abort();
      }
      return {
        ...state,
        loadingCombinedPatterns: false,
        loadingPatterns: false,
      };
    },
  },
  extraReducers: {
    [fetchDataset.fulfilled]: (state, action) => {
      let selectedGroupSetting = JSON.parse(
        JSON.stringify(state.selectedGroupSetting)
      );
      let selectedTheme = JSON.parse(JSON.stringify(state.selectedTheme));
      let element_to_label = {};
      let negative_element_to_label = {};
      let userAnnotationCount = 0;

      const data = action.payload;
      let dataset = [];
      let groups = [];
      let elements = {};

      data.forEach((element) => {
        dataset.push(element.id);
        elements[element.id] = element;
        if (element.user_label) {
          element_to_label[element.id] = element.user_label;
          const found =
            element.user_label &&
            selectedTheme &&
            element.user_label.find((el) => el == selectedTheme);
          if (found) {
            userAnnotationCount += 1;
          }
        }
        if (element.negative_user_label) {
          negative_element_to_label[element.id] = element.negative_user_label;
          // userAnnotationCount += 1;
        }
      });
      if (selectedGroupSetting == 0) {
        groups = [dataset];
      }
      return {
        ...state,
        dataset: dataset,
        elements: elements,
        ready: true,
        totalDataset: data.length,
        loading: false,
        groups: groups,
        element_to_label: element_to_label,
        negative_element_to_label: negative_element_to_label,
        userAnnotationCount: userAnnotationCount,
      };
    },
    [splitTheme.pending]: (state, action) => {
      return {
        ...state,
        loading: true,
      };
    },
    [splitTheme.fulfilled]: (state, action) => {
      const data = action.payload;

      let themes = data["all_themes"];

      let color_code = JSON.parse(JSON.stringify(state.color_code));
      let color_schema = JSON.parse(JSON.stringify(state.color_schema));

      let element_to_label = JSON.parse(JSON.stringify(state.element_to_label));
      let negative_element_to_label = JSON.parse(
        JSON.stringify(state.negative_element_to_label)
      );

      const old_theme = data["old_theme"];
      const new_themes = data["new_themes"];
      const pos_update_label = data["pos_update_labels"];
      const neg_update_label = data["neg_update_labels"];

      new_themes.forEach((theme) => {
        const pos_of_theme = pos_update_label[theme];

        pos_of_theme.forEach((elementId) => {
          //update postive element_to_level
          if (element_to_label[elementId]) {
            element_to_label[elementId] = element_to_label[elementId].filter(
              (arrayItem) => arrayItem != old_theme
            );
            element_to_label[elementId].push(theme);
          } else {
            element_to_label[elementId] = [theme];
          }
        });

        //update negative negative_element_to_level
        const neg_of_theme = neg_update_label[theme];

        neg_of_theme.forEach((elementId) => {
          //update postive element_to_level
          if (negative_element_to_label[elementId]) {
            negative_element_to_label[elementId] = negative_element_to_label[
              elementId
            ].filter((arrayItem) => arrayItem != old_theme);
            negative_element_to_label[elementId].push(theme);
          } else {
            negative_element_to_label[elementId] = [theme];
          }
        });
      });

      let index = Object.keys(color_code).length;

      let new_color_code = { ...color_code };

      themes.forEach((theme) => {
        if (color_code[theme]) {
          new_color_code[theme] = color_code[theme];
        } else {
          new_color_code[theme] = color_schema[index];
          index += 1;
        }
      });


      const userAnnotationCount = get_user_annotation_count(
        element_to_label,
        negative_element_to_label,
        data["selected_theme"]
      );
      return {
        ...state,
        loading: false,
        selectedTheme: data["selected_theme"],
        themes: data["all_themes"],
        color_code: new_color_code,
        explanation: {},
        element_to_label: element_to_label,
        negative_element_to_label: negative_element_to_label,
        userAnnotationCount: userAnnotationCount,
      };
    },
    [splitThemeByPattern.pending]: (state, action) => {
      return {
        ...state,
        loading: true,
      };
    },
    [splitThemeByPattern.fulfilled]: (state, action) => {
      return {
        ...state,
        loading: false,
      };
    },
    [toggleBinaryMode.fulfilled]: (state, action) => {},

    [mergeThemes.pending]: (state, action) => {
      return {
        ...state,
        loading: true,
      };
    },
    [mergeThemes.fulfilled]: (state, action) => {
      let data = action.payload;

      let color_code = JSON.parse(JSON.stringify(state.color_code));
      let color_schema = JSON.parse(JSON.stringify(state.color_schema));

      let element_to_label = JSON.parse(JSON.stringify(state.element_to_label));

      data["pos_update"].forEach((elementId) => {
        if (element_to_label[elementId])
          element_to_label[elementId].push(data["new_theme"]);
      });

      let index = Object.keys(color_code).length;

      let new_color_code = { ...color_code };

      data["all_themes"] &&
        data["all_themes"].forEach((theme) => {
          if (color_code[theme]) {
            new_color_code[theme] = color_code[theme];
          } else {
            new_color_code[theme] = color_schema[index];
            index += 1;
          }
        });


      return {
        ...state,
        themes: data["all_themes"],
        color_code: new_color_code,
        loading: false,
        element_to_label: element_to_label,
      };
    },
    [explainPattern.pending]: (state, action) => {
      const data = action.payload;
      return {
        ...state,
        patternExp: null,
      };
    },

    [explainPattern.fulfilled]: (state, action) => {
      const data = action.payload;


      Object.keys(data).map((value, index) => console.log(data[value][1]));

      return {
        ...state,
        patternExp: data,
      };
    },

    [fetchThemes.fulfilled]: (state, action) => {
      const data = action.payload;
      let color_schema = JSON.parse(JSON.stringify(state.color_schema));

      let color_code = {};
      data.forEach((element, index) => {
        color_code[`${element}`] = color_schema[index];
      });

      return {
        ...state,
        themes: data,
        color_code,
      };
    },

    [deleteTheme.pending]: (state, action) => {
      return {
        ...state,
        loading: true,
      };
    },

    [deleteTheme.fulfilled]: (state, action) => {
      const data = action.payload;

      let element_to_label = JSON.parse(JSON.stringify(state.element_to_label));

      let color_code = JSON.parse(JSON.stringify(state.color_code));
      let color_schema = JSON.parse(JSON.stringify(state.color_schema));

      let negative_element_to_label = JSON.parse(
        JSON.stringify(state.negative_element_to_label)
      );

      data["pos_update"] &&
        data["pos_update"].forEach((elementId) => {
          element_to_label[elementId] = element_to_label[elementId].filter(
            function (item) {
              return item !== data["deleted_theme"];
            }
          );
        });
      data["neg_update"] &&
        data["neg_update"].forEach((elementId) => {
          negative_element_to_label[elementId] = negative_element_to_label[
            elementId
          ].filter(function (item) {
            return item !== data["deleted_theme"];
          });
        });

      const remove_color = color_code[data["deleted_theme"]];

      color_schema = color_schema.filter((item) => item != remove_color);

      color_schema.push(remove_color);

      const userAnnotationCount = get_user_annotation_count(
        element_to_label,
        negative_element_to_label,
        data["selected_theme"]
      );
      return {
        ...state,
        loading: false,
        selectedTheme: data["selected_theme"],
        themes: data["all_themes"],
        loading: false,
        element_to_label: element_to_label,
        negative_element_to_label: negative_element_to_label,
        userAnnotationCount: userAnnotationCount,
        explanation: {},

        color_schema: color_schema,
      };
    },

    [fetchSelectedTheme.fulfilled]: (state, action) => {
      const data = action.payload;
      let userAnnotationCount = 0;
      //updateAnnotationCounter
      if (data) {
        const element_to_label = JSON.parse(
          JSON.stringify(state.element_to_label)
        );
        const negative_element_to_label = JSON.parse(
          JSON.stringify(state.negative_element_to_label)
        );

        userAnnotationCount = get_user_annotation_count(
          element_to_label,
          negative_element_to_label,
          data
        );
      }

      return {
        ...state,
        selectedTheme: data,
        userAnnotationCount: userAnnotationCount,
      };
    },

    [setTheme.pending]: (state, action) => {
      return {
        ...state,
        loading: true,
      };
    },

    [setTheme.fulfilled]: (state, action) => {
      const data = action.payload;
      let element_to_label = JSON.parse(JSON.stringify(state.element_to_label));
      let negative_element_to_label = JSON.parse(
        JSON.stringify(state.negative_element_to_label)
      );


      //update Annotation counter
      let userAnnotationCount = get_user_annotation_count(
        element_to_label,
        negative_element_to_label,
        data[0]
      );

      return {
        ...state,
        dataset: data[1],
        selectedTheme: data[0],
        explanation: {},
        combinedPatterns: {},
        patterns: [],
        userAnnotationCount: userAnnotationCount,
        modelAnnotationCount: 0,
        loading: false,
        cacheHit: true,
      };
    },

    [labelPhrase.pending]: (state, action) => {},

    [labelPhrase.fulfilled]: (state, action) => {
      const data = action.payload;
      if (data && data["positive"] == 0) {
        return {
          ...state,
        };
      }
      let id = data["id"];
      let phrase = data["phrase"];

      let label = data["label"];

      let elements = JSON.parse(JSON.stringify(state.elements));

      const start_index = elements[id]["example"].search(phrase);


      if (start_index == -1) {
        return {
          ...state,
        };
      }

      const spaces_within = phrase.split(" ").length - 1;
      const spaces_before =
        elements[id]["example"].substring(0, start_index).split(" ").length - 1;

      const spaces_after = elements[id]["example"]
        .substring(start_index + 1, elements[id]["example"].length)
        .split(" ").length;

      const total_spaces = elements[id]["example"].split(" ").length - 1;

      const start_highlight = total_spaces - spaces_after + 1;
      const end_highlight = start_highlight + spaces_within + 1;

      let explanation = JSON.parse(JSON.stringify(state.explanation));
      // console.log(explanation["USER_DEFINED"]);
      if (!explanation["USER_DEFINED"]) explanation["USER_DEFINED"] = {};

      if (!explanation["USER_DEFINED"][id])
        explanation["USER_DEFINED"][id] = [];

      explanation["USER_DEFINED"][id].push([
        elements[id]["example"]
          .split(" ")
          .slice(start_highlight, end_highlight),
        start_highlight,
        end_highlight,
        label,
      ]);


      return {
        ...state,
        explanation,
      };
    },

    [fetchPatterns.pending]: (state, action) => {
      let userAnnotationTracker = 0;

      return {
        ...state,
        loadingPatterns: true,
        userAnnotationTracker,
        refresh: false,
      };
    },

    [fetchPatterns.fulfilled]: (state, action) => {
      const data = action.payload;
      if (data.status_code == 300 || data.status_code == 500) {
        return {
          ...state,
          loadingPatterns: false,
          modelAnnotationCount: 0,

          combinedPatterns: {},
          patterns: [],
        };
      }
      //make rankby default and make group by none
      const selectedTheme = JSON.parse(JSON.stringify(state.selectedTheme));
      const groups = JSON.parse(JSON.stringify(state.groups));
      patterns_cache[selectedTheme] = data;
      return {
        ...state,
        patterns: data,
        loadingPatterns: false,
        cacheHit: true,
        backupGroups: groups,
      };
    },

    [fetchRelatedExample.fulfilled]: (state, action) => {
      const data = action.payload;

      return {
        ...state,
        relatedExamples: data[0],
        relatedExplanation: data[1],
      };
    },

    [fetchCombinedPatterns.pending]: (state, action) => {
      return {
        ...state,
        loadingCombinedPatterns: true,
      };
    },
    [fetchGroupedDataset.pending]: (state, action) => {
      return {
        ...state,
        loading: true,
      };
    },

    [fetchGroupedDataset.fulfilled]: (state, action) => {
      let data = action.payload;
      if (!data["data"]) {
        data["data"] = [JSON.parse(JSON.stringify(state.dataset))];
      }
      if (data["status_code"] == 300) {
        data["data"] = [JSON.parse(JSON.stringify(state.dataset))];
      }

      return {
        ...state,
        loading: false,
        groups: data["data"],
        backupGroups: data["data"],
        groupNames: data["group_names"],
      };
    },

    [fetchCombinedPatterns.fulfilled]: (state, action) => {
      const data = action.payload;

      if (data.message) {
        return {
          ...state,
          modelAnnotationCount: 0,
          loadingCombinedPatterns: false,
          loading: false,
        };
      }

      let modelAnnotationCount = 0;

      let selectedSetting = JSON.parse(JSON.stringify(state.selectedSetting));

      let patterns = JSON.parse(JSON.stringify(state.patterns));

      let dataset = JSON.parse(JSON.stringify(state.dataset));

      let elements = JSON.parse(JSON.stringify(state.elements));

      let reorderedGroups = JSON.parse(JSON.stringify(state.groups));

      for (const [key, value] of Object.entries(data["scores"])) {
        elements[key]["score"] = value;

        if (value != 0.5) modelAnnotationCount += 1;
      }


      reorderedGroups = reorderDataset(
        dataset,
        selectedSetting,
        elements,
        reorderedGroups
      );

      let selectedPatterns = {};
      data.patterns.forEach((element) => {
        selectedPatterns[element["pattern"]] = element["weight"];
        try {
          patterns[element["pattern"]]["status"] = 1;
        } catch {
          console.log("Caught Error: pattern doesn't exist");
        }
      });

      const selectedTheme = JSON.parse(JSON.stringify(state.selectedTheme));
      combinedPatterns_cache[selectedTheme] = data;
      explanations_cache[selectedTheme] = data.explanation;
      selectedPatterns_cache[selectedTheme] = selectedPatterns;
      patterns_cache[selectedTheme] = patterns;
      modelannotationCount_cache[selectedTheme] = modelAnnotationCount;

      const explanation = JSON.parse(JSON.stringify(state.explanation));
      let new_explanation = data.explanation || {};
      new_explanation["USER_DEFINED"] = explanation["USER_DEFINED"];

      return {
        ...state,
        combinedPatterns: data,
        loadingCombinedPatterns: false,
        explanation: new_explanation,
        dataset: dataset,
        patterns: patterns,
        elements: elements,
        modelAnnotationCount: modelAnnotationCount,
        selectedPatterns: selectedPatterns,
        groups: reorderedGroups,
      };
    },
  },
});

export default DataSlice.reducer;
export const {
  updateElementLabel,
  updateNegativeElementLabel,
  addTheme,
  updatePatExp,
  updateBinaryMode,
  changeSetting,
  changeGroupingSetting,
  updatePatterns,
  clearAnnotations,
  groupAnnotations,
  renameThemeLocal,
  clearHighlight,
  getCache,
  abortApiCall,
} = DataSlice.actions;
