import { Box } from "@material-ui/core";
import { Stack } from "@mui/material";
import * as React from "react";
import { useDispatch, useSelector } from "react-redux";
import TabelView from "../../sharedComponents/TableView";
import Header from "../HeaderComponent";
import Summary from "../../sharedComponents/Summary";
import { fetchDataset } from "./src/actions/dataset_actions";
import { labelPhrase } from "./src/actions/annotation_actions";
import { fetchSelectedTheme, setTheme } from "./src/actions/theme_actions";
import {
  labelData,
  fetchPatterns,
  fetchCombinedPatterns,
  fetchThemes,
  changeSetting,
} from "../../actions/Dataslice";
import Scroller from "../MarkedScrollbar";

import CustomPopover from "../../sharedComponents/CustomPopover";
import Popover from "@mui/material/Popover";
import ExplainPattern from "../../sharedComponents/TableView/explain_pattern";
import CustomLoading from "../../sharedComponents/Loading/Loading";
import SentenceViewer from "./SentenceViewer";

function Workspace() {
  //States
  const [focusedId, setFocusedId] = React.useState(0);
  const [labelCounter, setLabelCounter] = React.useState(0);
  const [hovering, setHovering] = React.useState(null);
  const [positiveIds, setPositiveIds] = React.useState({});
  const [scrollPosition, setScrollPosition] = React.useState(0);
  const [openSideBar, setOpenSideBar] = React.useState(false);
  const [loading, setLoading] = React.useState(true);

  const [activeSentenceGroup, setActiveSentenceGroup] = React.useState(0);

  const [labelSelectorAnchor, setLabelSelectorAnchor] = React.useState(null);

  const workspace = useSelector((state) => state.workspace);

  const dispatch = useDispatch();

  React.useEffect(() => {
    retrain();
  }, []);

  //Context menu
  const [anchorPoint, setAnchorPoint] = React.useState(null);
  const [showContextMenu, setShowContextMenu] = React.useState(false);
  const [popoverAnchor, setPopoverAnchor] = React.useState(null);
  const [popoverContent, setPopoverContent] = React.useState(null);
  const [openModal, setOpenModal] = React.useState(false);
  const [patternRow, setPatternRow] = React.useState(null);

  const getSelection = (event) => {
    let selected = window.getSelection().toString();

    if (selected.length == 0) {
      return;
    }
    event.preventDefault();

    setAnchorPoint(event.currentTarget);
    setShowContextMenu(true);
  };

  const handleBatchLabeling = () => {
    setPositiveIds({});
    setHovering(null);

    for (const [key, value] of Object.entries(positiveIds)) {
      dispatch(labelData({ element_id: key, label: value }));
    }
    dispatch(fetchPatterns()).then((response) => {
      const data = response.payload;
      if (data["status_code"] != 300) {
        dispatch(changeSetting({ selectedSetting: 0 }));
        dispatch(fetchCombinedPatterns());
      }
    });
  };

  const handleAddToPos = (elem) => {
    let ps = { ...positiveIds };
    ps[elem.element_id] = elem.label;
    setPositiveIds(ps);
  };

  const retrain = () => {
    if (labelCounter > 0 || true) {
      setLabelCounter(0);
      dispatch(fetchPatterns()).then((response) => {
        const data = response.payload;
        if (data["status_code"] != 300) {
          dispatch(changeSetting({ selectedSetting: 0 }));
          dispatch(fetchCombinedPatterns());
        }
      });
    }
  };

  const clear_session = () => {
    setLabelCounter(0);
    setHovering(null);
    setPositiveIds({});
    // setScrollPosition(0)
  };

  const handleChangeTheme = (value) => {
    clear_session();
    dispatch(setTheme({ theme: value })).then(() => {
      if (workspace.userAnnotationCount > 0 && workspace.refresh) retrain();
    });
    clear_session();
  };

  //Effects
  React.useEffect(() => {
    dispatch(fetchDataset());
    dispatch(fetchThemes());
    dispatch(fetchSelectedTheme());
  }, []);

  React.useEffect(() => {
    if (
      workspace.userAnnotationCount > 0 &&
      workspace.userAnnotationCount % workspace.annotationPerRetrain == 0
    ) {
      dispatch(fetchPatterns()).then((retrain) => {
        const data = response.payload;
        if (data["status_code"] != 300) {
          dispatch(changeSetting({ selectedSetting: 0 }));
          dispatch(fetchCombinedPatterns());
        }
      });
    }
  }, [workspace.userAnnotationCount]);

  const filterHovering = (hovering) => {
    let filteredDataset = [];
    const exp = workspace.explanation[hovering];
  };


  const handleOpenModal = (row) => {
    setPatternRow(row);
    setOpenModal(true);
  };

  return (
    <Stack direction={"column"} sx={{ height: "100vh" }}>
      <CustomLoading />
      <Header
        binary_mode={workspace.binary_mode}
        setTheme={handleChangeTheme}
        color_code={workspace.color_code}
        selectedTheme={workspace.selectedTheme}
        themes={workspace.themes}
        retrain={retrain}
        annotationPerRetrain={workspace.annotationPerRetrain}
        userAnnotationTracker={workspace.userAnnotationTracker}
        modelAnnotationCount={workspace.modelAnnotationCount}
        totalDataset={workspace.totalDataset}
        userAnnotationCount={workspace.userAnnotationCount}
      />
      <Scroller
        color={workspace.color_code[workspace.selectedTheme]}
        dataset={workspace.dataset}
        scrollPosition={scrollPosition}
        show={!hovering}
        activeSentenceGroup={activeSentenceGroup}
      />
      <Stack direction={"row"} sx={{ height: "84vh" }} mt={"145px"} ml={1}>
        <SentenceViewer
          positiveIds={positiveIds}
          setPositiveIds={setPositiveIds}
          hovering={hovering}
          setHovering={setHovering}
          setScrollPosition={setScrollPosition}
          setOpenSideBar={setOpenSideBar}
          handleAddToPos={handleAddToPos}
          focusedId={focusedId}
          setFocusedId={setFocusedId}
          handleBatchLabeling={handleBatchLabeling}
          setPopoverAnchor={setPopoverAnchor}
          setPopoverContent={setPopoverContent}
          setLabelSelectorAnchor={setLabelSelectorAnchor}
          getSelection={getSelection}
          setActiveSentenceGroup={setActiveSentenceGroup}
        />

        <Box
          style={{
            maxHeight: "100%",
            maxWidth: "50vw",
            minWidth: "50vw",
            overflow: "auto",
          }}
        >
          <Stack direction={"column"}>
            <Summary
              selectedTheme={workspace.selectedTheme}
              color={workspace.color_code[workspace.selectedTheme]}
              data={workspace.combinedPatterns}
              retrain={retrain}
            />
            <TabelView
              index={0}
              setHovering={setHovering}
              hovering={hovering}
              handelOpenModal={handleOpenModal}
              loading={workspace.loadingCombinedPatterns}
              data={workspace.combinedPatterns.patterns}
              columns={[
                { id: "pattern", label: "Pattern" },
                { id: "weight", label: "Weight" },
                { id: "fscore", label: "Fscore" },
                { id: "recall", label: "Recall" },
                { id: "precision", label: "Precision" },
              ]}
            />

            <TabelView
              index={1}
              handelOpenModal={handleOpenModal}
              loading={workspace.loadingPatterns}
              data={workspace.patterns}
              columns={[
                { id: "pattern", label: "Pattern" },
                { id: "fscore", label: "Fscore" },
                { id: "recall", label: "Recall" },
                { id: "precision", label: "Precision" },
              ]}
            />
          </Stack>
        </Box>

        <CustomPopover
          open={showContextMenu}
          anchorPoint={anchorPoint}
          handleClose={(label) => {
            let selected = window.getSelection().toString();

            setShowContextMenu(false);
            setAnchorPoint(null);
          }}
        />
      </Stack>

      <Popover
        id={"generaluse_popover"}
        anchorEl={popoverAnchor}
        open={Boolean(popoverAnchor)}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "left",
        }}
        sx={{
          pointerEvents: "none",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "left",
        }}
        disableRestoreFocus
        children={
          <>
            <Box
              sx={{
                position: "relative",
                mt: "10px",
                "&::before": {
                  backgroundColor: "white",
                  content: '""',
                  display: "block",
                  position: "absolute",
                  width: 12,
                  height: 12,
                  top: -6,
                  transform: "rotate(45deg)",
                  left: "calc(50% - 6px)",
                },
              }}
            />
            <Box sx={{ p: 2, backgroundColor: "white" }}>{popoverContent}</Box>
          </>
        }
        PaperProps={{
          style: {
            backgroundColor: "transparent",
            borderRadius: 1,
          },
        }}
      />

      <ExplainPattern
        open={openModal}
        setOpen={setOpenModal}
        setRow={setPatternRow}
        row={patternRow}
      />
    </Stack>
  );
}

export default {
  routeProps: {
    path: "/",
    element: (
        <Workspace />
    ),
  },
  name: "workspace",
};
