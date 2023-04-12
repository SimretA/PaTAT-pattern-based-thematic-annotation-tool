import * as React from "react";
import Backdrop from "@mui/material/Backdrop";
import Box from "@mui/material/Box";
import Modal from "@mui/material/Modal";
import Fade from "@mui/material/Fade";
import Typography from "@mui/material/Typography";
import { Stack, Chip, Button } from "@mui/material";
import TextField from "@mui/material/TextField";
import ArrowCircleLeftIcon from "@mui/icons-material/ArrowCircleLeft";
import ArrowCircleRightIcon from "@mui/icons-material/ArrowCircleRight";
import IconButton from "@mui/material/IconButton";
import SentenceLight from "../Sentence/sentenceLight";
import { useDispatch, useSelector } from "react-redux";
import { fetchUserlabeledData } from "../../actions/dataset_actions";
import { renameThemeRemote, splitThemeByPattern } from "../../actions/theme_actions";
import {
  
  clearAnnotations,
  renameThemeLocal,
  
} from "../../actions/Dataslice";
import { CircularProgress, Divider } from "@mui/material";
import { lighten } from "@material-ui/core";
import Fab from "@mui/material/Fab";
import { CSS_COLOR_NAMES } from "../../assets/color_assets";
import Menu from "@mui/material/Menu";
import MenuItem from "@mui/material/MenuItem";
import { deleteTheme, mergeThemes, splitTheme } from "../../actions/theme_actions";
import { TransferWithinAStationTwoTone } from "@mui/icons-material";
import CustomLoading from "../Loading/Loading";

const style = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: 900,

  bgcolor: "background.paper",
  border: "1px solid #cccccc",
  boxShadow: 24,
  p: 4,
};

export default function SplitThemeModal({
  open,
  setOpen,
  theme,
  action,
  actionType,
  retrain,
}) {
  const workspace = useSelector((state) => state.workspace);

  const dispatch = useDispatch();

  const [anchorEl, setAnchorEl] = React.useState(null);
  const [sentences, setSentences] = React.useState([{}, {}]);
  const [patterns, setPatterns] = React.useState([[], []]);
  const [groupNames, setGroupNames] = React.useState([``, ``]);
  const [patternsToSplit, setPatternsToSplit] = React.useState({});
  const [splitByPattern, setSplitByPattern] = React.useState(false);
  const [splitStage, setSplitStage] = React.useState(null);
  const [loadingLeft, setLoadingLeft] = React.useState(false);
  const [loadingRight, setLoadingRight] = React.useState(false);

  const [renameTheme, setRenameTheme] = React.useState(theme);

  const [mergedName, setMergeName] = React.useState("");
  const [defaultMergeName, setDefaultMergeName] = React.useState("");
  const [nextColor, setNextColor] = React.useState(null);

  React.useEffect(() => {
    if (actionType == "merge") {
      setDefaultMergeName(`${groupNames[0]}_${groupNames[1]}`);
    }
  }, [groupNames[0], groupNames[1]]);

  React.useEffect(() => {
    setRenameTheme(theme);
  }, [theme]);
  const handleDeleteTheme = (selectedTheme) => {
    dispatch(deleteTheme({ theme: selectedTheme })).then(() => {
      dispatch(clearAnnotations());
      retrain();
    });
  };
  const handleSplitTheme = (selectedTheme, group1, group2) => {
    dispatch(splitTheme({ theme: selectedTheme, group1, group2 })).then(
      (response) => {
      }
    );
  };

  const getExamples = () => {
    dispatch(
      splitThemeByPattern({
        theme: workspace.selectedTheme,
        patterns: patterns[1],
        new_theme_name: groupNames[1],
      })
    ).then((response) => {
      setLoadingLeft(false);
      setSentences([response.payload["group1"], response.payload["group2"]]);
    });
  };

  const handleAction = (type) => {
    switch (type) {
      case "delete":
        handleDeleteTheme(workspace.selectedTheme);
        handleClose();
        break;
      case "split":
        updateGrouping();
        handleClose();
        break;
      case "rename":
        if (renameTheme.trim() != "") {
          dispatch(
            renameThemeLocal({
              theme: workspace.selectedTheme,
              new_name: renameTheme.trim(),
            })
          );

          dispatch(
            renameThemeRemote({
              theme: workspace.selectedTheme,
              new_name: renameTheme.trim(),
            })
          );
          handleClose();
        }
        break;
      case "merge":
        if (groupNames[0] && groupNames[1] && groupNames[1].trim() != "") {
          dispatch(
            mergeThemes({
              theme1: workspace.selectedTheme,
              theme2: groupNames[1],
              new_theme:
                mergedName.trim() == "" ? defaultMergeName : mergedName,
            })
          ).then((response) => {
            setDefaultMergeName("");
            setMergeName("");
          });
          handleClose();
        }
        break;
      default:
      // code block
    }
  };

  const setName = (value, index) => {
    let new_groupNames = [...groupNames];
    new_groupNames[index] = value;
    setGroupNames(new_groupNames);
  };
  React.useEffect(() => {
    if (actionType == "merge" || actionType == "split") {
      setLoadingLeft(true);
      setDefaultMergeName(`${groupNames[0]}`);
    }
  }, [actionType]);
  React.useEffect(() => {
    if (open) {
      setNextColor(workspace.color_schema[workspace.themes.length]);
      if (actionType == "split") {
        if (Object.keys(workspace.explanation).length <= 1) {
          setSplitStage(1);
          dispatch(
            fetchUserlabeledData({ theme: workspace.selectedTheme })
          ).then((stuff) => {
            setLoadingLeft(false);
            setSentences([stuff.payload, {}]);
          });
        } else {
          setSplitStage(0);
          setPatterns([Object.keys(workspace.explanation), []]);
        }
        setGroupNames([workspace.selectedTheme, ""]);
      } else if (actionType == "delete") {
        setGroupNames(["hello", "there"]);
      } else if (actionType == "merge") {
        setLoadingLeft(true);
        setGroupNames([`${workspace.selectedTheme}`, ""]);

        dispatch(fetchUserlabeledData({ theme: workspace.selectedTheme })).then(
          (stuff) => {
            let new_sentences = sentences;
            new_sentences[0] = stuff.payload;
            setSentences([...new_sentences]);
            setLoadingLeft(false);
          }
        );
      }
    }
  }, [actionType, open]);

  const handleClose = () => {
    setOpen(false);
    setSentences([{}, {}]);
    setSplitStage(null);
  };

  const setSecondGroupToMerge = (theme) => {
    let new_group_names = groupNames;
    new_group_names[1] = theme;
    setGroupNames(new_group_names);
    setAnchorEl(null);
    dispatch(fetchUserlabeledData({ theme: theme })).then((response) => {
      setLoadingRight(false);
      let new_sentences = sentences;
      if (new_sentences) {
        new_sentences[1] = response.payload;
      } else {
        new_sentences = [{}, response.payload];
      }

      setSentences([...new_sentences]);
    });
  };
  const updateGrouping = () => {
    let group1 = { name: groupNames[0], data: sentences[0] };

    let group2 = { name: groupNames[1], data: sentences[1] };

    handleSplitTheme(workspace.selectedTheme, group1, group2);
    handleClose();
    dispatch(clearAnnotations());
    retrain();
  };

  const SplitByPatternAction = () => {
    dispatch(
      splitThemeByPattern({
        theme: workspace.selectedTheme,
        patterns: Object.keys(patternsToSplit),
        new_theme_name: groupNames[1],
      })
    ).then((response) => {
      dispatch(clearAnnotations());
      retrain();
      handleClose();
    });
  };
  const handlePatternRegroup = (source, target, index) => {
    let new_patterns = [...patterns];

    let removed = new_patterns[source].splice(index, 1);

    new_patterns[target].push(removed[0]);

    setPatterns(new_patterns);
  };
  const handleRegroup = (source, target, sentenceId) => {
    let new_sentences = [...sentences];

    let temp_sentence = [...new_sentences[source][sentenceId]];

    new_sentences[target][sentenceId] = temp_sentence;

    delete new_sentences[source][sentenceId];

    setSentences(new_sentences);
  };

  return (
    <Modal
      aria-labelledby="transition-modal-title"
      aria-describedby="transition-modal-description"
      open={open}
      onClose={handleClose}
      closeAfterTransition
      BackdropComponent={Backdrop}
      BackdropProps={{
        timeout: 500,
      }}
      sx={{ overflow: "scroll" }}
    >
      <Fade in={open}>
        <Box
          sx={{
            ...style,
            overflow: "hidden",
          }}
        >
          <Typography id="transition-modal-title" variant="h6" component="h2">
            Updating{" "}
            <Chip
              key={`menuitem_${workspace.selectedTheme}_light`}
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
            {workspace.selectedTheme} - {actionType}
          </Typography>
          <Divider />
          {actionType == "split" && (
            <Stack
              id="transition-modal-description"
              sx={{
                mt: 2,
                position: "relative",
                overflow: "hidden",

                height: "70vh",
              }}
              direction="column"
              spacing={5}
            >
              <>
                <Stack direction={"row"} style={{ minHeight: "50vh" }}>
                  <Stack
                    divider={<Divider orientation="horizontal" flexItem />}
                    id="transition-modal-description"
                    direction="column"
                    width={450}
                    spacing={5}
                    sx={{
                      maxHeight: "50vh",
                      overflow: "scroll",
                    }}
                  >
                    <Stack
                      sx={{
                        position: "sticky",
                        top: "0px",
                        backgroundColor: "#FFFFFF",
                        zIndex: 100,
                      }}
                    >
                      <TextField
                        id="standard-basic"
                        variant="outlined"
                        value={groupNames[0]}
                        onChange={(event) => {
                          setName(event.target.value.toLowerCase(), 0);
                        }}
                        InputProps={{
                          startAdornment: (
                            <Chip
                              label={""}
                              color={"primary"}
                              sx={{
                                backgroundColor:
                                  workspace.color_code[workspace.selectedTheme],
                                width: 20,
                                height: 20,
                                marginRight: 1,
                              }}
                              size="small"
                            />
                          ),
                        }}
                      />
                    </Stack>
                    {splitStage == 0 ? (
                      <>
                        {patterns &&
                          patterns[0].map((pattern, index) => (
                            <Stack direction={"row"}>
                              <Typography>{pattern}</Typography>
                              <IconButton
                                onClick={() => {
                                  handlePatternRegroup(0, 1, index);
                                }}
                              >
                                <ArrowCircleRightIcon />
                              </IconButton>
                            </Stack>
                          ))}
                      </>
                    ) : (
                      <>
                        {loadingLeft && splitStage == 1 && (
                          <>
                            <CircularProgress />
                          </>
                        )}
                        {sentences &&
                          sentences[0] &&
                          Object.keys(sentences[0]).length > 0 &&
                          Object.keys(sentences[0]).map((key, index) => (
                            <Stack
                              direction={"row"}
                              key={`groupsent_stack_${key}`}
                            >
                              <SentenceLight
                                color={
                                  workspace.color_code[workspace.selectedTheme]
                                    ? lighten(
                                        workspace.color_code[
                                          workspace.selectedTheme
                                        ],
                                        0.5
                                      )
                                    : "blue"
                                }
                                show={true}
                                // highlight={workspace.relatedExplanation[element.id]}
                                element={workspace.elements[key]}
                                handleBatchLabel={null}
                                sentence={sentences[0][key][0]}
                                key={`groupsent_${key}`}
                              />

                              {/* <Typography>{sentences[0][key][0]}</Typography> */}
                              <IconButton
                                onClick={() => {
                                  handleRegroup(0, 1, key);
                                }}
                              >
                                <ArrowCircleRightIcon />
                              </IconButton>
                            </Stack>
                          ))}
                      </>
                    )}
                  </Stack>

                  <Stack
                    divider={<Divider orientation="horizontal" flexItem />}
                    id="transition-modal-description"
                    direction="column"
                    width={450}
                    spacing={5}
                    sx={{ maxHeight: "50vh", overflow: "scroll" }}
                  >
                    <Stack
                      sx={{
                        position: "sticky",
                        top: "0px",
                        backgroundColor: "#FFFFFF",
                        zIndex: 100,
                      }}
                    >
                      <TextField
                        placeholder={"Theme"}
                        id="standard-basic"
                        variant="outlined"
                        value={groupNames[1]}
                        onChange={(event) => {
                          setName(event.target.value.toLowerCase(), 1);
                        }}
                        InputProps={{
                          startAdornment: (
                            <Chip
                              label={""}
                              color={"primary"}
                              sx={{
                                backgroundColor: nextColor,
                                width: 20,
                                height: 20,
                                marginRight: 1,
                              }}
                              size="small"
                            />
                          ),
                        }}
                      />
                    </Stack>
                    {splitStage == 0 ? (
                      <>
                        {patterns[1].map((pattern, index) => (
                          <Stack direction={"row"}>
                            <IconButton
                              onClick={() => {
                                handlePatternRegroup(1, 0, index);
                              }}
                            >
                              <ArrowCircleLeftIcon />
                            </IconButton>
                            <Typography>{pattern}</Typography>
                          </Stack>
                        ))}
                      </>
                    ) : (
                      <>
                        {sentences &&
                          sentences[1] &&
                          Object.keys(sentences[1]).length > 0 &&
                          Object.keys(sentences[1]).map((key, index) => (
                            <Stack
                              direction={"row"}
                              key={`groupsent_stack_${key}`}
                            >
                              <IconButton
                                onClick={() => {
                                  handleRegroup(1, 0, key);
                                }}
                              >
                                <ArrowCircleLeftIcon />
                              </IconButton>
                              <SentenceLight
                                color={
                                  workspace.color_code[workspace.selectedTheme]
                                    ? lighten(
                                        workspace.color_code[
                                          workspace.selectedTheme
                                        ],
                                        0.5
                                      )
                                    : `${lighten("#ececec", 0.5)}`
                                }
                                show={true}
                                // highlight={workspace.relatedExplanation[element.id]}
                                element={workspace.elements[key]}
                                handleBatchLabel={null}
                                sentence={sentences[1][key][0]}
                                key={`groupsent_${key}`}
                              />
                            </Stack>
                          ))}
                      </>
                    )}
                  </Stack>
                </Stack>

                <Stack direction={"row"} spacing={5}>
                  {Object.keys(workspace.explanation).length > 1 && (
                    <Fab
                      variant="extended"
                      m={10}
                      color={"primary"}
                      onClick={() => {
                        if (splitStage == 0) {
                          setSplitStage(1);
                          //get examples with this grouping
                          getExamples();
                        } else setSplitStage(0);
                      }}
                    >
                      {splitStage == 0 ? "Next" : "Back"}
                    </Fab>
                  )}

                  <Fab
                    variant="extended"
                    m={10}
                    color={"primary"}
                    onClick={() => updateGrouping()}
                    disabled={
                      !sentences ||
                      (sentences &&
                        (Object.keys(sentences[0]).length == 0 ||
                          Object.keys(sentences[1]).length == 0)) ||
                      groupNames[0].trim() == "" ||
                      groupNames[1].trim() == ""
                    }
                  >
                    Update
                  </Fab>
                </Stack>
              </>
            </Stack>
          )}

          {actionType == "rename" && (
            <Stack direction={"row"} spacing={5} m={5}>
              <TextField
                inputRef={(input) => input && input.focus()}
                placeholder={"Theme"}
                value={renameTheme}
                onChange={(event) =>
                  setRenameTheme(event.target.value.toLowerCase())
                }
              />
              <Button
                variant="contained"
                onClick={() => {
                  handleAction(actionType);
                }}
              >
                Rename
              </Button>
            </Stack>
          )}
          {actionType == "delete" && (
            <Stack spacing={3} mt={3}>
              <Typography
                id="transition-modal-title"
                variant="body2"
                gutterBottom
              >
                Are you sure you want to delete{" "}
                <Chip
                  label={""}
                  color={"primary"}
                  sx={{
                    backgroundColor:
                      workspace.color_code[workspace.selectedTheme],
                    width: 20,
                    height: 20,
                    marginRight: 1,
                  }}
                  size="small"
                />{" "}
                {workspace.selectedTheme}?
              </Typography>
              <Stack direction={"row"} spacing={5}>
                <Button
                  size="small"
                  sx={{ backgroundColor: "#bdbdbd" }}
                  variant={"contained"}
                  onClick={() => handleClose()}
                >
                  Cancel
                </Button>
                <Button
                  size="small"
                  color={"error"}
                  variant={"contained"}
                  onClick={() => {
                    handleAction(actionType);
                  }}
                >
                  Delete
                </Button>
              </Stack>
            </Stack>
          )}

          {actionType == "merge" && (
            <>
              <Stack
                sx={{
                  position: "sticky",
                  top: "0px",
                  backgroundColor: "#FFFFFF",
                  zIndex: 100,
                }}
              >
                <TextField
                  id="standard-basic"
                  variant="outlined"
                  value={mergedName}
                  placeholder={defaultMergeName}
                  onChange={(event) => {
                    setMergeName(event.target.value.toLowerCase());
                  }}
                  InputProps={{
                    startAdornment: (
                      <Chip
                        label={""}
                        color={"primary"}
                        sx={{
                          backgroundColor: nextColor,
                          width: 20,
                          height: 20,
                          marginRight: 1,
                        }}
                        size="small"
                      />
                    ),
                  }}
                />
              </Stack>
              <Stack
                id="transition-modal-description"
                sx={{
                  mt: 2,
                  overflow: "hidden",
                  position: "relative",
                  height: "50vh",
                }}
                direction="row"
                spacing={5}
                divider={<Divider orientation="vertical" flexItem />}
              >
                <Stack
                  divider={<Divider orientation="horizontal" flexItem />}
                  id="transition-modal-description"
                  direction="column"
                  width={450}
                  spacing={5}
                  sx={{ maxHeight: "80vh", overflow: "scroll" }}
                >
                  <Typography
                    id="transition-modal-title"
                    variant="h6"
                    component="h2"
                    sx={{
                      position: "sticky",
                      top: "0px",
                      backgroundColor: "#FFFFFF",
                      zIndex: 5,
                    }}
                  >
                    <Chip
                      label={""}
                      color={"primary"}
                      sx={{
                        backgroundColor: workspace.color_code[groupNames[0]],
                        width: 20,
                        height: 20,
                        marginRight: 1,
                      }}
                      size="small"
                    />{" "}
                    {groupNames[0]}
                  </Typography>
                  {sentences &&
                    sentences[0] &&
                    Object.keys(sentences[0]).length > 0 &&
                    Object.keys(sentences[0]).map((key, index) => (
                      <Stack direction={"row"} key={`groupsent_stack_${key}`}>
                        <SentenceLight
                          color={
                            workspace.color_code[workspace.selectedTheme]
                              ? lighten(
                                  workspace.color_code[workspace.selectedTheme],
                                  0.5
                                )
                              : `${lighten("#ececec", 0.5)}`
                          }
                          show={true}
                          // highlight={workspace.relatedExplanation[element.id]}
                          element={workspace.elements[key]}
                          handleBatchLabel={null}
                          sentence={sentences[0][key][0]}
                          key={`groupsent_${key}`}
                        />
                      </Stack>
                    ))}

                  {loadingLeft && (
                    <>
                      <CircularProgress />
                    </>
                  )}
                </Stack>

                <Stack
                  id="transition-modal-description"
                  direction="column"
                  width={450}
                  spacing={5}
                  sx={{ maxHeight: "80vh", overflow: "scroll" }}
                >
                  <Typography
                    id="transition-modal-title"
                    variant="h6"
                    component="h2"
                    sx={{
                      cursor: "pointer",
                      position: "sticky",
                      top: "0px",
                      backgroundColor: "#FFFFFF",
                      zIndex: 5,
                    }}
                    onClick={(event) => {
                      setAnchorEl(event.target);
                    }}
                  >
                    <Chip
                      key={`menuitem_${groupNames[1]}_light`}
                      label={""}
                      color={"primary"}
                      sx={{
                        backgroundColor: workspace.color_code[groupNames[1]],
                        width: 20,
                        height: 20,
                        marginRight: 1,
                      }}
                      size="small"
                    />{" "}
                    {groupNames[1].trim().length > 0
                      ? groupNames[1]
                      : "Select Theme"}
                  </Typography>
                  <Menu
                    id="long-menu"
                    MenuListProps={{
                      "aria-labelledby": "long-button",
                    }}
                    anchorEl={anchorEl}
                    open={Boolean(anchorEl)}
                    onClose={() => setAnchorEl(null)}
                    PaperProps={{
                      style: {
                        width: "20ch",
                      },
                    }}
                  >
                    {workspace.themes.map((theme) => (
                      <MenuItem
                        key={theme}
                        disabled={theme == workspace.selectedTheme}
                        onClick={(event) => {
                          setLoadingRight(true);
                          setSecondGroupToMerge(theme);
                        }}
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
                        />
                        {theme}
                      </MenuItem>
                    ))}
                  </Menu>
                  <Divider orientation="horizontal" flexItem />
                  {sentences &&
                    sentences[1] &&
                    Object.keys(sentences[1]).length > 0 &&
                    Object.keys(sentences[1]).map((key, index) => (
                      <Stack direction={"row"} key={`groupsent_stack_${key}`}>
                        <SentenceLight
                          color={
                            workspace.color_code[workspace.selectedTheme]
                              ? lighten(
                                  workspace.color_code[workspace.selectedTheme],
                                  0.5
                                )
                              : `${lighten("#ececec", 0.5)}`
                          }
                          show={true}
                          // highlight={workspace.relatedExplanation[element.id]}
                          element={workspace.elements[key]}
                          handleBatchLabel={null}
                          sentence={"sentences[1][key][0]"}
                          key={`groupsent_${key}`}
                        />
                      </Stack>
                    ))}

                  {loadingRight && (
                    <>
                      <CircularProgress />
                    </>
                  )}
                </Stack>
              </Stack>
              <Divider orientation="horizontal" flexItem />
              <Button
                sx={{ marginTop: 5 }}
                onClick={() => handleAction(actionType)}
                variant={"contained"}
              >
                Merge
              </Button>
            </>
          )}
        </Box>
      </Fade>
    </Modal>
  );
}
