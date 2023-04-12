import * as React from "react";
import Typography from "@mui/material/Typography";
import { Chip, Stack, Button, CircularProgress, Divider } from "@mui/material";
import Highlight from "./Highlight";
import { useDispatch, useSelector } from "react-redux";
import { multiLabelData, 
  groupAnnotationsRemote,
   deleteLabelData } from "../../actions/annotation_actions";

   import { fetchRelatedExample } from "../../actions/pattern_actions";
import {
  updateElementLabel,
  updateNegativeElementLabel,
  groupAnnotations,  
  clearHighlight,
} from "../../actions/Dataslice";
import SentenceLight from "./sentenceLight";
import { useEffect } from "react";
import AddIcon from "@mui/icons-material/Add";
import { lighten } from "@material-ui/core";

export default function AccordionSentence(props) {
  const dispatch = useDispatch();
  const workspace = useSelector((state) => state.workspace);

  const [matchedParts, setMatchedParts] = React.useState(null);
  const [matchedIndex, setMatchedIndex] = React.useState(null);
  const [sentence, setSentence] = React.useState(null);
  const [expandMore, setExpandMore] = React.useState(false);

  const [hidePrediction, setHidePrediction] = React.useState(false);
  const [showAcceptorReject, setShowAcceptorReject] = React.useState(true);

  const [labeled, setLabeled] = React.useState(
    workspace.userLabel[props.elementId]
  );
  const [loading, setLoading] = React.useState(false);
  const [activateSelection, setActivateSelection] = React.useState(false);

  React.useEffect(() => {
    setSentence(props.example);
  }, []);

  React.useEffect(() => {
    setHidePrediction(false);
    setShowAcceptorReject(true);
  }, [props.score]);

  // useEffect(() => {
  //   if (workspace.binary_mode) {
  //     setShowAcceptorReject(false);
  //   }
  // }, [workspace.binary_mode]);
  useEffect(() => {
    if (props.score > 0.5 || true) {
      setMatchedParts(getMatchedParts(props.elementId, props.example));
    } else {
      let parts = {};
      parts[`${props.example}`] = [false, 0, -1, []];
      setMatchedParts(parts);
    }
  }, [props.score, workspace.explanation]);

  const handleShowMore = () => {
    if (expandMore) {
      setExpandMore(false);
      props.retrain();
    } else {
      setExpandMore(true);
      setLoading(true);

      dispatch(fetchRelatedExample({ id: props.elementId })).then(() => {
        setLoading(false);
      });
    }
  };

  const handleBatchLabel = (element_id, label) => {
    props.setPositiveIds({ element_id, label });
  };

  const getMatchedParts = (id, sentence) => {
    const explanations = workspace.explanation;
    let user_defined = explanations["USER_DEFINED"];
    let sentence_array = sentence.split(" ");
    let matchedParts = {};
    sentence_array.forEach((element, index) => {
      matchedParts[index] = [];
    });

    for (let [key, value] of Object.entries(explanations)) {
      const exp = value && value[id];
      if (key == "USER_DEFINED" && exp) {
        value[id].forEach((explanation) => {
          const start = explanation[1];
          const end = explanation[2];

          for (let i = start; i < end; i++) {
            try {
              matchedParts[i] = [...matchedParts[i], key];
            } catch (error) {
            }
          }
        });
      } else if (exp && exp != "") {
        const start = value[id][0][1];
        const end = value[id][0][2];

        for (let i = start; i < end; i++) {
          try {
            matchedParts[i] = [...matchedParts[i], key];
          } catch (error) {
          }
        }
      }
    }
    setMatchedIndex(matchedParts);

    let highlighted = {};

    let substr = "";
    let current = null;
    let start_index_counter = 0;
    let end_index_counter = 0;
    let previous_patterns = [];
    let current_patterns = [];

    for (let [key, value] of Object.entries(matchedParts)) {
      let matched = value.length > 0;
      if (current == null) {
        current = matched;
        substr += " " + sentence_array[key];
        previous_patterns = value;

        current_patterns = [...value];
      } else {
        if (current != matched) {
          highlighted[`${substr} ${key}`] = [
            current,
            start_index_counter,
            end_index_counter,
            current_patterns,
          ];

          start_index_counter = end_index_counter;

          current = matched;
          substr = sentence_array[key];

          current_patterns = [...value];
        } else {
          const filteredArray = previous_patterns.filter((value2) =>
            value.includes(value2)
          );
          if (filteredArray.length) {
            substr += " " + sentence_array[key];
            current_patterns = [...new Set([...current_patterns, ...value])];
          } else {
            highlighted[`${substr} ${key}`] = [
              current,
              start_index_counter,
              end_index_counter,
              current_patterns,
            ];

            start_index_counter = end_index_counter;

            current = matched;
            current_patterns = [...value];
            substr = sentence_array[key];
          }
        }

        if (sentence_array.length - 1 == key) {
          highlighted[`${substr} ${key}`] = [
            current,
            start_index_counter,
            end_index_counter,
            current_patterns,
          ];
        }
      }

      end_index_counter += 1;
      previous_patterns = value;
    }

    return highlighted;
  };

  const handleGroupLabeling = (label, groupIndex, positive, elementId) => {
    let ids = workspace.groups[groupIndex].filter((id) => id != elementId);

    dispatch(groupAnnotations({ ids: ids, label: label, positive: positive }));
    dispatch(
      groupAnnotationsRemote({ ids: ids, label: label, positive: positive })
    ).then((response) => {
      props.retrain();
    });
  };

  const getSentenceParts = (sentence_list, start, end) => {
    let sentence = [];
    if (start > 0) {
      sentence.push([sentence_list.slice(0, start).join(" "), 0]);
    }
    sentence.push([sentence_list.slice(start, end).join(" "), 1]);
    if (end < sentence_list.length - 1) {
      sentence.push([
        sentence_list.slice(end, sentence_list.length).join(" "),
        0,
      ]);
    }

    return sentence;
  };

  const clear_highlight = (key, label, start, end, patterns) => {
    let matched = { ...matchedParts };
    matched[key][0] = false;
    setMatchedParts(matched);

    dispatch(
      clearHighlight({
        key,
        label,
        start,
        end,
        patterns,
        elementId: props.elementId,
      })
    );
  };

  const handleAddLabel = (event, label) => {
    props.setAnchorEl(event.currentTarget);
    // setSentenceLabels([...sentenceLabels,"Hello"])
  };

  const handelDeleteLabel = (elementId, label) => {
    dispatch(updateElementLabel({ elementId, label, event: "REMOVE" }));
    dispatch(deleteLabelData({ elementId, label }));
  };

  const handleAcceptOrReject = (elementId, score, accept) => {
    setShowAcceptorReject(false);
    if (accept == 1) {

      if (score > 0.5) {
        dispatch(
          updateElementLabel({
            elementId: elementId,
            label: workspace.selectedTheme,
            event: "ADD",
          })
        );
        dispatch(
          multiLabelData({
            elementId: elementId,
            label: workspace.selectedTheme,
            positive: 1,
          })
        );
      } else if (score < 0.5) {
        dispatch(
          updateNegativeElementLabel({
            elementId: elementId,
            theme: workspace.selectedTheme,
            label: 0,
          })
        );

        dispatch(
          multiLabelData({
            elementId: elementId,
            label: workspace.selectedTheme,
            positive: 0,
          })
        );
      }
    } else if (accept == 0) {
      setHidePrediction(true);

      if (score > 0.5) {
        dispatch(
          updateNegativeElementLabel({
            elementId: elementId,
            theme: workspace.selectedTheme,
            label: 0,
          })
        );

        dispatch(
          multiLabelData({
            elementId: elementId,
            label: workspace.selectedTheme,
            positive: 0,
          })
        );
      } else if (score < 0.5) {
        dispatch(
          updateElementLabel({
            elementId: elementId,
            label: workspace.selectedTheme,
            event: "ADD",
          })
        );

        dispatch(
          multiLabelData({
            elementId: elementId,
            label: workspace.selectedTheme,
            positive: 1,
          })
        );
      }
    }
  };
  const findIt = (label) => {
    return label == workspace.selectedTheme;
  };

  return (
    <Stack
      elevation={1}
      onClick={() => {
        props.setFocusedId(props.elementId);
      }}
      sx={{
        marginLeft: 0,
        padding: 2,
        backgroundColor: "#FFFFFF",
        border: "solid 3px #ececec",
        zIndex: 1,
        ...(expandMore && { position: "sticky", top: 0 }),
      }}
    >
      <Stack direction="column">
        <Stack direction="row">
          <Stack spacing={1} direction={"column"}>
            {props.score != null && props.score != 0.5 && !hidePrediction && (
              <Stack
                style={{
                  justifyContent: "center",
                  alignItems: "center",
                  color:
                    props.score < 0.5
                      ? "crimson"
                      : workspace.color_code[workspace.selectedTheme],
                  border:
                    props.score < 0.5
                      ? "1px solid crimson"
                      : `1px solid ${
                          workspace.color_code[workspace.selectedTheme]
                        }`,
                  borderRadius: 13.125,
                  padding: 5,
                  marginRight: 5,
                  marginBottom: 5,
                }}
                color={props.score > 0.5 ? "success" : "error"}
              >
                <Typography
                  style={{
                    maxWidth: "100px",
                    flexWrap: "wrap",
                    fontSize: "14px",
                  }}
                >
                  {props.score > 0.5
                    ? `Predicted  ${props.theme}`
                    : `Predicted not ${props.theme}`}
                </Typography>
                {!(
                  workspace.element_to_label[props.elementId] &&
                  workspace.element_to_label[props.elementId].indexOf(
                    workspace.selectedTheme
                  ) != -1
                ) &&
                  showAcceptorReject &&
                  !workspace.binary_mode && (
                    <Stack
                      direction={"row"}
                      justifyContent="space-evenly"
                      alignItems="center"
                      spacing={0}
                      maxWidth={"fit-content"}
                    >
                      <Button
                        onClick={() => {
                          handleAcceptOrReject(props.elementId, props.score, 1);
                        }}
                        color="info"
                        size="small"
                        style={{
                          border: "1px solid",
                          borderLeft: "none",
                          borderRadius: "20px 0px 0px 20px",
                          fontSize: "12px",
                        }}
                      >
                        Accept
                      </Button>
                      <Button
                        onClick={() => {
                          handleAcceptOrReject(props.elementId, props.score, 0);
                        }}
                        size="small"
                        style={{
                          border: "1px solid",
                          borderLeft: "none",
                          borderRadius: "0px 20px 20px 0px",
                          fontSize: "12px",
                        }}
                        color="info"
                      >
                        Reject
                      </Button>
                    </Stack>
                  )}
              </Stack>
            )}
          </Stack>

          <Typography
            sx={{ fontSize: 16, fontWeight: 500 }}
            color="text.secondary"
            align="left"
            variant="body2"
            display="block"
            gutterBottom
            onMouseUp={(event) => props.getSelection(event)}
          >
            <Stack
              direction={"row"}
              spacing={1}
              maxWidth={"100%"}
              overdflowY={"visible"}
              overflowX={"hidden"}
              sx={{ flexWrap: "wrap" }}
            >
              <Typography
                sx={{
                  fontSize: 16,
                  fontWeight: 500,
                }}
                color="text.secondary"
                align="left"
                variant="body2"
                display="block"
                gutterBottom
              >
                <Stack
                  direction={"row"}
                  spacing={1}
                  maxWidth={"100%"}
                  sx={{ flexWrap: "wrap", zIndex: 10, userSelect: "text" }}
                >
                  {matchedParts &&
                    Object.keys(matchedParts).map((key, index) => (
                      <Highlight
                        elementId={props.elementId}
                        key={`sent_${props.elementId}_${index}`}
                        score={props.score}
                        word={key}
                        matched={matchedParts[key][0]}
                        deleteMatched={clear_highlight}
                        start={matchedParts[key][1]}
                        end={matchedParts[key][2]}
                        patterns={matchedParts[key][3]}
                        matchedWith={matchedIndex}
                        setPopoverAnchor={props.setPopoverAnchor}
                        setPopoverContent={props.setPopoverContent}
                      />
                    ))}

                  {!matchedParts && <Highlight word={props.example} />}
                  {/* {props.element.example} */}
                </Stack>
              </Typography>
            </Stack>
          </Typography>
        </Stack>
      </Stack>

      <div>
        <Stack
          direction={"row"}
          spacing={2}
          style={{ flexWrap: "wrap", rowGap: "10px" }}
        >
          {workspace.element_to_label[props.elementId] &&
            [...new Set(workspace.element_to_label[props.elementId])].map(
              (label) => (
                <Chip
                  key={`${props.elementId}_${label}`}
                  label={label}
                  color={"primary"}
                  sx={{ backgroundColor: workspace.color_code[label] }}
                  size="small"
                  onDelete={() => handelDeleteLabel(props.elementId, label)}
                />
              )
            )}
          {workspace.negative_element_to_label[props.elementId] &&
            [
              ...new Set(workspace.negative_element_to_label[props.elementId]),
            ].map((negative_label) => (
              <Chip
                key={`${props.elementId}_neg_${negative_label}`}
                label={`Not ${negative_label}`}
                color={"primary"}
                sx={{ backgroundColor: workspace.not_color }}
                style={{
                  border: `5px solid ${workspace.color_code[negative_label]}`,
                }}
                size="small"
              />
            ))}

          {workspace.binary_mode ? (
            <Stack
              direction="row"
              spacing={1}
              ml={5}
              style={{
                border: `1px solid ${
                  workspace.color_code[workspace.selectedTheme]
                }`,
                padding: "5px",
                borderRadius: "2px",
              }}
            >
              <Typography variant="subtitle1">
                {`${workspace.selectedTheme}?`}
              </Typography>
              <Button
                variant={"outlined"}
                size={"small"}
                backgroundColor={workspace.color_code[workspace.selectedTheme]}
                disabled={
                  workspace.element_to_label[props.elementId] &&
                  workspace.element_to_label[props.elementId].findIndex(
                    findIt
                  ) != -1
                }
                onClick={() => {
                  dispatch(
                    updateNegativeElementLabel({
                      elementId: props.elementId,
                      theme: workspace.selectedTheme,
                      label: 1,
                    })
                  );
                  dispatch(
                    multiLabelData({
                      elementId: props.elementId,
                      label: workspace.selectedTheme,
                      positive: 1,
                    })
                  );
                }}
              >
                Yes
              </Button>

              <Button
                disabled={
                  workspace.negative_element_to_label[props.elementId] &&
                  workspace.negative_element_to_label[
                    props.elementId
                  ].findIndex(findIt) != -1
                }
                onClick={() => {
                  dispatch(
                    updateNegativeElementLabel({
                      elementId: props.elementId,
                      theme: workspace.selectedTheme,
                      label: 0,
                    })
                  );
                  dispatch(
                    multiLabelData({
                      elementId: props.elementId,
                      label: workspace.selectedTheme,
                      positive: 0,
                    })
                  );
                }}
                variant={"outlined"}
                size={"small"}
                color={"error"}
              >
                No
              </Button>
            </Stack>
          ) : (
            <Chip
              icon={<AddIcon />}
              label="Add Label"
              variant="outlined"
              size="small"
              onClick={(event) => handleAddLabel(event, props.elementId)}
            />
          )}

          {props.score > 0.5 && (
            <Button onClick={() => handleShowMore()}>
              {expandMore ? "Done" : "See Similar"}
            </Button>
          )}
        </Stack>

        {expandMore && (
          <Stack
            direction={"column"}
            maxHeight={"50vw"}
            overflow={"auto"}
            style={{
              border: "solid 3px #ececec",
              borderTop: "0px",
              zIndex: 1111,
              backgroundColor: "#FFFFFF",
            }}
          >
            <Divider />

            {loading && (
              <CircularProgress
                sx={{ position: "fixed", top: "30%", left: "300px" }}
              ></CircularProgress>
            )}

            {!loading &&
              workspace.relatedExamples.map((element) => (
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
                  highlight={workspace.relatedExplanation[element.id]}
                  element={element}
                  handleBatchLabel={handleBatchLabel}
                  sentence={element.example}
                  key={`lightsent_${element.id}`}
                />
              ))}
          </Stack>
        )}
      </div>
    </Stack>
  );
}
