import * as React from "react";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import { useDispatch, useSelector } from "react-redux";
import { multiLabelData, groupAnnotationsRemote } from "../../actions/annotation_actions";
import {
  updateElementLabel,
  updateNegativeElementLabel,
  addTheme,
  groupAnnotations,
} from "../../actions/Dataslice";
import { setTheme, addThemeRemote } from "../../actions/theme_actions";
import { Chip } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import Button from "@mui/material/Button";

import TextField from "@mui/material/TextField";
import { Box } from "@material-ui/core";

export default function LabelSelector({
  anchorEl,
  setAnchorEl,
  elementId,
  merge,
  groupLabeling,
  setBatchLabeling,
}) {
  const workspace = useSelector((state) => state.workspace);

  const [addNewTheme, setAddNewTheme] = React.useState(false);

  const dispatch = useDispatch();


  const open = Boolean(anchorEl);

  const handleAddNewTheme = (event) => {
    if (event.key === "Enter" && event.target.value.trim() != "") {
      //dispatch add new theme
      dispatch(
        addTheme({ theme: event.target.value, index: workspace.themes.length })
      );
      dispatch(addThemeRemote({ theme: event.target.value }));
      if (workspace.selectedTheme == null) {
        dispatch(setTheme({ theme: event.target.value }));
      }

      //Add element label
      handleAddLabel(event.target.value);
      setAddNewTheme(false);
    }
  };

  const handleClose = () => {
    setAnchorEl(null);
    setBatchLabeling(null);
  };
  const handleAddLabel = (label) => {
    setAnchorEl(null);
    if (groupLabeling) {
      let ids = groupLabeling.filter((id) => id != elementId);
      setBatchLabeling(null);

      dispatch(groupAnnotations({ ids: ids, label: label, positive: 1 }));
      dispatch(
        groupAnnotationsRemote({ ids: ids, label: label, positive: 1 })
      ).then((response) => {
        // props.retrain();
      });
    } else {
      // workspace.element_to_label[elementId]= [label]
      dispatch(updateElementLabel({ elementId, label, event: "ADD" }));
      dispatch(multiLabelData({ elementId, label }));
    }
  };

  const handleAddBinaryLabel = (elementId, theme, label) => {
    setAnchorEl(null);
    if (groupLabeling) {
      let ids = groupLabeling.filter((id) => id != elementId);
      setBatchLabeling(null);

      dispatch(groupAnnotations({ ids: ids, label: theme, positive: label }));
      dispatch(
        groupAnnotationsRemote({ ids: ids, label: theme, positive: label })
      ).then((response) => {
        // props.retrain();
      });
    } else {
      dispatch(updateNegativeElementLabel({ elementId, theme, label }));
      dispatch(multiLabelData({ elementId, label: theme, positive: label }));
    }
  };

  const findIt = (label) => {
    return label == workspace.selectedTheme;
  };
  return !workspace.binary_mode ? (
    <Menu
      id="long-menu"
      MenuListProps={{
        "aria-labelledby": "long-button",
      }}
      anchorEl={anchorEl}
      open={open}
      onClose={handleClose}
      PaperProps={{
        style: {
          // width: "20ch",
        },
      }}
    >
      {workspace.themes.map((theme) => (
        <MenuItem
          // sx={{ textTransform: "capitalize" }}
          key={theme}
          disabled={false}
          onClick={() => handleAddLabel(theme)}
        >
          <Chip
            key={`menuitem_${theme}_light`}
            label={""}
            color={"primary"}
            sx={{
              backgroundColor: workspace.color_code[theme],
              width: 20,
              height: 20,
              marginRight: 1,
            }}
            size="small"
          />{" "}
          {theme}
        </MenuItem>
      ))}
      <MenuItem
        size="small"
        onKeyDown={(e) => e.stopPropagation()}
        onClickCapture={(e) => {
          e.stopPropagation();
          if (!addNewTheme) setAddNewTheme(!addNewTheme);
        }}
        // sx={{ textTransform: "capitalize" }}
        value={"add_new_theme"}
        key={`theme_add_new`}
      >
        {addNewTheme ? (
          <Box
            sx={{
              display: "flex",
              alignItems: "flex-end",
              width: "100%",
            }}
          >
            <Chip
              key={`menuitem_new_theme_light${workspace.themes.length}`}
              label={""}
              color={"primary"}
              sx={{
                backgroundColor:
                  workspace.color_schema[workspace.themes.length],
                width: 20,
                height: 20,
                marginRight: 1,
                mr: 1,
                my: 0.5,
              }}
              size="small"
            />
            <TextField
              inputRef={(input) => input && input.focus()}
              size="small"
              onKeyDown={handleAddNewTheme}
              id="input-with-sx"
              label="New Theme"
              variant="standard"
              onChange={(event) => {
                event.target.value = event.target.value.toLowerCase();
              }}
              placeholder={"Theme"}
              multiline
              rowsMax="3"
              fullWidth={true}
            />
          </Box>
        ) : (
          <Button variant="outlined" startIcon={<AddIcon />}>
            Add Theme
          </Button>
        )}
      </MenuItem>
    </Menu>
  ) : (
    <Menu
      id="long-menu"
      MenuListProps={{
        "aria-labelledby": "long-button",
      }}
      anchorEl={anchorEl}
      open={open}
      onClose={handleClose}
      PaperProps={{
        style: {
          width: "20ch",
        },
      }}
    >
      <MenuItem
        // sx={{ textTransform: "capitalize" }}
        disabled={
          workspace.element_to_label[elementId] &&
          workspace.element_to_label[elementId].findIndex(findIt) != -1
        }
        onClick={() =>
          handleAddBinaryLabel(elementId, workspace.selectedTheme, 1)
        }
      >
        <Chip
          label={""}
          color={"primary"}
          sx={{
            backgroundColor: workspace.color_code[workspace.selectedTheme],
            width: 20,
            height: 20,
            marginRight: 1,
          }}
          size="small"
        />{" "}
        {workspace.selectedTheme}
      </MenuItem>

      <MenuItem
        // sx={{ textTransform: "capitalize" }}
        disabled={
          workspace.negative_element_to_label[elementId] &&
          workspace.negative_element_to_label[elementId].findIndex(findIt) != -1
        }
        onClick={() =>
          handleAddBinaryLabel(elementId, workspace.selectedTheme, 0)
        }
      >
        <Chip
          label={""}
          color={"primary"}
          sx={{
            backgroundColor: workspace.not_color,
            width: 20,
            height: 20,
            marginRight: 1,
          }}
          size="small"
        />{" "}
        not {workspace.selectedTheme}
      </MenuItem>
    </Menu>
  );
}
