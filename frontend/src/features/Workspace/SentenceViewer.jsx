import { Box, Typography } from "@material-ui/core";
import * as React from "react";
import { useDispatch, useSelector } from "react-redux";
import AccordionSentence from "../../sharedComponents/Sentence/sentence";
import { multiLabelData } from "../../actions/annotation_actions";
import {
  changeSetting,
  updateNegativeElementLabel,
} from "../../actions/Dataslice";
import { fetchCombinedPatterns, fetchPatterns } from "../../actions/pattern_actions";
import { Chip } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import { lighten } from "@material-ui/core";

import Fab from "@mui/material/Fab";
import LabelSelector from "../../sharedComponents/LabelSelector/labelselector";

import SentenceLight from "../../sharedComponents/Sentence/sentenceLight";
import { Stack } from "@mui/material";

export default function SentenceViewer({
  getSelection,
  hovering,
  setHovering,
  setScrollPosition,
  setOpenSideBar,
  focusedId,
  setFocusedId,
  setPopoverAnchor,
  setPopoverContent,
  setActiveSentenceGroup,
}) {
  const workspace = useSelector((state) => state.workspace);

  const dispatch = useDispatch();

  const [positiveIds, setPositiveIds] = React.useState({});

  const [labelSelectorAnchor, setLabelSelectorAnchor] = React.useState(null);
  const [batchLabeling, setBatchLabeling] = React.useState(null);

  React.useEffect(() => {
    if (workspace.loadingCombinedPatterns || workspace.loadingPatterns) {
      setHovering(null);
    }
  }, [workspace.loadingCombinedPatterns, workspace.loadingPatterns]);

  const handleBatchLabeling = () => {
    // setPositiveIds({});
    setHovering(null);

    for (const [key, value] of Object.entries(positiveIds)) {
      dispatch(
        multiLabelData({
          elementId: key,
          label: workspace.selectedTheme,
          positive: value,
        })
      );

      if (value == 1) {
        dispatch(
          updateNegativeElementLabel({
            elementId: key,
            theme: workspace.selectedTheme,
            label: 1,
          })
        );
      } else if (value == 0) {
        dispatch(
          updateNegativeElementLabel({
            elementId: key,
            theme: workspace.selectedTheme,
            label: 0,
          })
        );
      }
    }

    if (Object.keys(positiveIds).length > 0) {
      dispatch(fetchPatterns()).then((response) => {
        const data = response.payload;
        if (data["status_code"] != 300) {
          dispatch(changeSetting({ selectedSetting: 0 }));
          dispatch(fetchCombinedPatterns());
        }
      });
      setPositiveIds({});
    }
  };

  const handleAddToPos = (elem) => {
    let ps = { ...positiveIds };
    ps[elem.element_id] = elem.label;
    setPositiveIds(ps);
  };
  return (
    <Box
      style={{
        maxHeight: "100%",
        maxWidth: "50vw",
        minWidth: "50vw",
        overflow: "auto",
        marginLeft: "10px",
      }}
      onScroll={(event) => {
        setScrollPosition(event.target.scrollTop / event.target.scrollHeight);
      }}
    >
      <LabelSelector
        anchorEl={labelSelectorAnchor}
        setAnchorEl={setLabelSelectorAnchor}
        elementId={focusedId}
        groupLabeling={batchLabeling}
        setBatchLabeling={setBatchLabeling}
      />

      {!hovering &&
        workspace.groups.map((groups, groupIndex) => (
          <Stack
            mb={5}
            sx={{
              ...(workspace.groups.length > 1 && {
                border: "solid 8px #4f4a50",
                borderRadius: "10px",
              }),
            }}
          >
            {workspace.groups.length > 1 && !workspace.binary_mode && (
              <Stack
                direction={"row"}
                spacing={2}
                justifyContent="center"
                alignItems="center"
              >
                {workspace.groupNames && (
                  <Typography variant="h6">
                    {workspace.groupNames[groupIndex]}
                  </Typography>
                )}
                <Chip
                  icon={<AddIcon />}
                  label="Add Group Label"
                  variant="outlined"
                  size="small"
                  onClick={(event) => {
                    setBatchLabeling(workspace.groups[groupIndex]);
                    setLabelSelectorAnchor(event.target);
                  }}
                />
                {/* <Button
                  variant="contained"
                  size="small"
                  sx={{
                    maxWidth: "100px",
                    backgroundColor: "#cececece",
                    border: "solid 8px #4f4a50",
                  }}
                  onClick={(event) => {
                    setBatchLabeling(workspace.groups[groupIndex]);
                    setLabelSelectorAnchor(event.target);
                  }}
                >
                  Label Group
                </Button> */}
              </Stack>
            )}
            {groups.map((elementId, index) => (
              <AccordionSentence
                groupIndex={groupIndex}
                marginLeft={10}
                seeMore={setOpenSideBar}
                index={index}
                positiveIds={positiveIds}
                setPositiveIds={handleAddToPos}
                explanation={
                  hovering && workspace.explanation
                    ? workspace.explanation[hovering][elementId]
                    : null
                }
                hovering={hovering}
                score={workspace.elements[elementId].score}
                key={`sent_grouped_${elementId}`}
                elementId={elementId}
                example={workspace.elements[elementId].example}
                focused={focusedId == elementId}
                setFocusedId={setFocusedId}
                theme={workspace.selectedTheme}
                annotationPerRetrain={workspace.annotationPerRetrain}
                getSelection={getSelection}
                retrain={handleBatchLabeling}
                setPopoverAnchor={setPopoverAnchor}
                setPopoverContent={setPopoverContent}
                setAnchorEl={setLabelSelectorAnchor}
              />
            ))}
          </Stack>
        ))}

      {hovering &&
        workspace.explanation &&
        workspace.groups &&
        workspace.groups.map((groups, groupIndex) => (
          <Stack
            mb={10}
            sx={{ backgroundColor: "#cececece", border: "solid 3px #cececece" }}
          >
            {groups.map((elementId, index) => (
              <SentenceLight
                show={
                  workspace.explanation[hovering] &&
                  workspace.explanation[hovering][elementId]
                }
                highlight={
                  workspace.explanation[hovering] &&
                  workspace.explanation[hovering][elementId]
                }
                color={
                  workspace.color_code[workspace.selectedTheme]
                    ? lighten(
                        workspace.color_code[workspace.selectedTheme],
                        0.5
                      )
                    : "none"
                }
                element={workspace.elements[elementId]}
                handleBatchLabel={(element_id, label) =>
                  handleAddToPos({ element_id, label })
                }
                highlightAsList={true}
                sentence={workspace.elements[elementId].example}
                key={`lightsent_hovering_${elementId}`}
              />
            ))}
          </Stack>
        ))}

      {hovering && Object.keys(positiveIds).length > 0 && (
        <Fab
          sx={{ position: "sticky", bottom: "50px", marginLeft: "20px" }}
          color={"primary"}
          variant="extended"
          onClick={handleBatchLabeling}
        >
          Done
        </Fab>
      )}
    </Box>
  );
}
