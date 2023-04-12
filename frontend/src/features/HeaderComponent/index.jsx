import {
  AppBar,
  Typography,
  FormControlLabel,
  Switch,
} from "@material-ui/core";
import React from "react";
import { Stack, Chip, IconButton } from "@mui/material";
import Settings from "../../sharedComponents/Menu/index";
import GroupingSettings from "../../sharedComponents/Menu/grouping_setting";
import CircularProgress from "@mui/material/CircularProgress";
import Divider from "@mui/material/Divider";
import { Box } from "@material-ui/core";
import ProgressButton from "../../sharedComponents/ProgressButton";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import FormControl from "@mui/material/FormControl";
import Select from "@mui/material/Select";
import AddIcon from "@mui/icons-material/Add";
import Button from "@mui/material/Button";
import LogoutIcon from "@mui/icons-material/Logout";
import TextField from "@mui/material/TextField";
import { useDispatch, useSelector } from "react-redux";
import { useNavigate } from "react-router-dom";

import { lighten } from "@material-ui/core";
import { toggleBinaryMode } from "../../actions/annotation_actions";
import {
  addTheme,
  clearAnnotations,
  updateBinaryMode,
} from "../../actions/Dataslice";
import { addThemeRemote, setTheme } from "../../actions/theme_actions";
export default function Header(props) {
  const dispatch = useDispatch();

  const workspace = useSelector((state) => state.workspace);

  const navigate = useNavigate();

  const [addNewTheme, setAddNewTheme] = React.useState(false);
  const [userAnnotationPercent, setUserAnnotationPercent] = React.useState(0);
  const [modelAnnotationPercent, setModelAnnotationPercent] = React.useState(0);

  React.useState(0);
  const [nextColor, setNextColor] = React.useState(null);

  React.useEffect(() => {
    setNextColor(workspace.color_schema[workspace.themes.length]);
  }, [workspace.themes]);

  React.useEffect(() => {
    if (props.userAnnotationCount && props.totalDataset > 0) {
      setUserAnnotationPercent(
        (props.userAnnotationCount / props.totalDataset) * 100
      );
    }
  }, [props.userAnnotationCount]);
  React.useEffect(() => {
    if (workspace.modelAnnotationCount != null && props.totalDataset > 0) {
      setModelAnnotationPercent(
        (workspace.modelAnnotationCount / props.totalDataset) * 100
      );
    }
  }, [props.modelAnnotationCount, workspace.modelAnnotationCount]);

  const handleAddNewTheme = (event) => {
    if (event.key === "Enter" && event.target.value.trim() != "") {
      //dispatch add new theme
      dispatch(
        addTheme({ theme: event.target.value, index: props.themes.length })
      );
      dispatch(addThemeRemote({ theme: event.target.value }));
      if (workspace.selectedTheme == null) {
        dispatch(setTheme({ theme: event.target.value }));
      }
      setAddNewTheme(false);
    }
  };

  const handleThemeChange = (event) => {
    if (event.target.value == "add_new_theme") {
      event.preventDefault();
      setAddNewTheme(!addNewTheme);
    } else {
      props.setTheme(event.target.value);
      dispatch(clearAnnotations());
    }
  };
  const handleChangeBinaryMode = () => {
    dispatch(updateBinaryMode());
    dispatch(toggleBinaryMode({ binary_mode: props.binary_mode }));
  };

  return (
    <AppBar ml={2} color="inherit" maxHeight={"140px"}>
      <Box
        maxHeight={"35px"}
        sx={{ backgroundColor: "#000000", display: "inline" }}
        width={"100vw"}
      >
        <Typography
          variant="h5"
          style={{
            display: "inline",
            fontFamily: "Indie Flower",
            color: "#FFFFFF",
            width: "0px",
          }}
        >
          PaTAT
        </Typography>
      </Box>

      <Stack
        direction={"row"}
        ml={2}
        justifyContent="center"
        spacing={1}
        divider={<Divider orientation="vertical" flexItem />}
        alignItems="center"
      >
        <Settings />

        <GroupingSettings />

        <ProgressButton
          color={props.color_code[props.selectedTheme]}
          retrain={props.retrain}
          userAnnotationTracker={props.userAnnotationTracker}
          annotationPerRetrain={props.annotationPerRetrain}
          value={
            ((props.userAnnotationCount % props.annotationPerRetrain) /
              props.annotationPerRetrain) *
              100 || 0
          }
        />

        <Stack
          sx={{
            alignContent: "center",
            alignItems: "center",
            justifyContent: "center",
            flexGrow: 1,
          }}
          ml={2}
          mr={2}
          direction={"row"}
          spacing={1}
        >
          {/* <Typography >Progress:</Typography> */}

          <Stack
            p={2}
            sx={{
              alignContent: "center",
              alignItems: "center",
              justifyContent: "center",
            }}
            ml={2}
            mr={2}
            direction={"column"}
          >
            {/* <CircularProgress thickness={10} variant="determinate" value={(props.userAnnotationCount/props.totalDataset)*100} /> */}

            <Box sx={{ position: "relative", display: "inline-flex" }}>
              <CircularProgress
                thickness={20}
                variant="determinate"
                value={userAnnotationPercent}
                //backgroundColor:`${lighten('#86de8c', 1-props.score)}
                sx={{
                  color:
                    props.selectedTheme && props.color_code[props.selectedTheme]
                      ? lighten(props.color_code[props.selectedTheme], 0.5)
                      : "#FFFFFF",
                }}
              />
              <Box
                sx={{
                  top: 0,
                  left: 0,
                  bottom: 0,
                  right: 0,
                  position: "absolute",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Typography variant="caption" component="div">
                  {`${Math.round(userAnnotationPercent)}%`}
                </Typography>
              </Box>
            </Box>

            <Typography variant="caption" color={"textSecondary"}>
              You
            </Typography>
          </Stack>

          <Stack
            p={2}
            sx={{
              alignContent: "center",
              alignItems: "center",
              justifyContent: "center",
            }}
            ml={2}
            mr={2}
            direction={"column"}
          >
            <Box sx={{ position: "relative", display: "inline-flex" }}>
              <CircularProgress
                thickness={20}
                variant="determinate"
                value={modelAnnotationPercent}
                sx={{
                  color:
                    props.selectedTheme && props.color_code[props.selectedTheme]
                      ? lighten(props.color_code[props.selectedTheme], 0.5)
                      : "#FFFFFF",
                }}
              />
              <Box
                sx={{
                  top: 0,
                  left: 0,
                  bottom: 0,
                  right: 0,
                  position: "absolute",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <Typography variant="caption" component="div">
                  {`${Math.round(modelAnnotationPercent)}%`}
                </Typography>
              </Box>
            </Box>
            <Typography variant="caption" color={"textSecondary"}>
              Model
            </Typography>
          </Stack>
        </Stack>

        <Stack p={2} sx={{ minWidth: "200px" }}>
          {/* <Typography>{props.selectedTheme}</Typography> */}

          <FormControl fullWidth>
            <InputLabel id="demo-simple-select-label">Theme</InputLabel>
            <Select
              labelId="demo-simple-select-label"
              id="demo-simple-select"
              value={props.selectedTheme ? props.selectedTheme : ""}
              label="Theme"
              onChange={(event) => handleThemeChange(event)}
              displayEmpty
            >
              {props.themes.map((theme, index) => (
                <MenuItem
                  size="small"
                  // sx={{ textTransform: "capitalize" }}
                  value={theme}
                  key={`theme_${theme}`}
                >
                  <Chip
                    key={`menuitem_${theme}_light`}
                    label={""}
                    color={"primary"}
                    sx={{
                      backgroundColor: props.color_code[theme],
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
                  if (!addNewTheme) {
                    setNextColor(
                      workspace.color_schema[workspace.themes.length]
                    );

                    setAddNewTheme(!addNewTheme);
                  }
                }}
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
                      key={`menuitem_new_theme_light${props.themes.length}`}
                      label={""}
                      color={"primary"}
                      sx={{
                        backgroundColor: nextColor,
                        width: 20,
                        height: 20,
                        marginRight: 1,
                        mr: 1,
                        my: 0.5,
                      }}
                      size="small"
                    />
                    <TextField
                      size="small"
                      onKeyDown={handleAddNewTheme}
                      id="input-with-sx"
                      label="New Theme"
                      variant="standard"
                      inputRef={(input) => input && input.focus()}
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
            </Select>
          </FormControl>
        </Stack>

        <FormControlLabel
          control={
            <Stack direction="row" spacing={1} alignItems="center">
              <Typography>Multi-label</Typography>
              <Switch
                disabled={
                  workspace.themes.length == 0 ||
                  workspace.selectedTheme == null
                }
                style={{ color: props.color_code[props.selectedTheme] }}
                value={props.binary_mode}
                onChange={() => {
                  handleChangeBinaryMode();
                }}
              />
              <Typography>Binary</Typography>
            </Stack>
          }
        />
      </Stack>
    </AppBar>
  );
}
