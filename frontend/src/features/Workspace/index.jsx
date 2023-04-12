import { Box, Divider } from "@material-ui/core";
import { Stack } from "@mui/material";
import * as React from "react";
import { useDispatch, useSelector } from "react-redux";
import TabelView from "../../sharedComponents/TableView";
import Header from "../HeaderComponent";
import Summary from "../../sharedComponents/Summary";

import { fetchDataset, createSession } from "../../actions/dataset_actions";
import { labelPhrase } from "../../actions/annotation_actions";
import { fetchSelectedTheme, fetchThemes, setTheme } from "../../actions/theme_actions";
import {
  changeSetting,
  getCache,
  abortApiCall,
} from "../../actions/Dataslice";
import { fetchCombinedPatterns, fetchPatterns } from "../../actions/pattern_actions";
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
  const [scrollPosition, setScrollPosition] = React.useState(0);
  const [openSideBar, setOpenSideBar] = React.useState(false);

  const [activeSentenceGroup, setActiveSentenceGroup] = React.useState(0);

  const [labelSelectorAnchor, setLabelSelectorAnchor] = React.useState(null);

 

  const workspace = useSelector((state) => state.workspace);

  const dispatch = useDispatch();

  // React.useEffect(() => {
  //   retrain();
  // }, []);

  //Context menu
  const [anchorPoint, setAnchorPoint] = React.useState(null);
  const [x, setX] = React.useState(0);
  const [y, setY] = React.useState(0);
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
    setX(event.clientX);
    setY(event.clientY);
    setShowContextMenu(true);
  };
  React.useEffect(() => {
    if (workspace.cacheHit == false) {
      if (workspace.loadingPatterns || workspace.loadingCombinedPatterns) {
        dispatch(abortApiCall());
      }
      retrain();
    }
  }, [workspace.cacheHit]);

  const retrain = () => {
    if (labelCounter > 0 || true) {
      setLabelCounter(0);
      dispatch(fetchPatterns()).then((response) => {
        const data = response.payload;

        if (!data || data["status_code"] != 300) {
          dispatch(fetchCombinedPatterns());
        }
      });
    }
  };

  const clear_session = () => {
    setLabelCounter(0);
    setHovering(null);
    // setPositiveIds({});
    // setScrollPosition(0)
  };

  const handleChangeTheme = (value) => {
    clear_session();
    dispatch(setTheme({ theme: value })).then(() => {
      if (workspace.userAnnotationCount > 0 && workspace.refresh) retrain();
      else {
        dispatch(getCache());
      }
    });
    clear_session();
  };

  //Effects
  React.useEffect(() => {
    dispatch(createSession({ user: "user" })).then((response) => {
      window.localStorage.setItem("user", "user")
      dispatch(fetchDataset());
      dispatch(fetchThemes());
      dispatch(fetchSelectedTheme());
    });
  }, []);

  React.useEffect(() => {
    if (
      workspace.userAnnotationCount > 0 &&
      workspace.userAnnotationTracker >= workspace.annotationPerRetrain
    ) {
      dispatch(fetchPatterns()).then((response) => {
        const data = response.payload;
        if (!data || data["status_code"] != 300) {
          dispatch(changeSetting({ selectedSetting: 0 }));
          dispatch(fetchCombinedPatterns());
        }
      });
    }
  }, [workspace.userAnnotationTracker]);

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
      />
      <Stack
        direction={"row"}
        sx={{ height: "84vh" }}
        mt={"145px"}
        ml={1}
        spacing={2}
        divider={<Divider orientation="vertical" />}
      >
        <SentenceViewer
          hovering={hovering}
          setHovering={setHovering}
          setScrollPosition={setScrollPosition}
          setOpenSideBar={setOpenSideBar}
          focusedId={focusedId}
          setFocusedId={setFocusedId}
          setPopoverAnchor={setPopoverAnchor}
          setPopoverContent={setPopoverContent}
          setLabelSelectorAnchor={setLabelSelectorAnchor}
          getSelection={getSelection}
          setActiveSentenceGroup={setActiveSentenceGroup}
        />

        <Box
          style={{
            maxHeight: "100%",
            maxWidth: "45vw",
            minWidth: "45vw",
            overflow: "auto",
            padding: "10px",
            paddingTop: "0px",
          }}
        >
          <Stack direction={"column"}>
            <Stack
              style={{
                position: "sticky",
                top: 0,
                minWidth: "100%",
                backgroundColor: "#FFFFFF",
                zIndex: 1000,
                boxShadow: "2px 2px",
              }}
            >
              <Summary
                selectedTheme={workspace.selectedTheme}
                color={workspace.color_code[workspace.selectedTheme]}
                data={workspace.combinedPatterns}
                retrain={retrain}
              />
            </Stack>
            <TabelView
              index={0}
              setHovering={setHovering}
              hovering={hovering}
              handelOpenModal={handleOpenModal}
              loading={
                workspace.loadingCombinedPatterns || workspace.loadingPatterns
              }
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
              data={Object.values(workspace.patterns)}
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
          handleClose={() => {
            setShowContextMenu(false);
            setAnchorPoint(null);
          }}
          x={x}
          y={y}
          handlePhraseLabeling={(label) => {
            let selected = window.getSelection().toString().replace(/\n/g, " ");
            if (selected.trim() != "") {
              dispatch(
                labelPhrase({
                  phrase: selected,
                  label: label,
                  id: focusedId,
                  positive: 1,
                })
              ).then(() => {
                window.getSelection().empty();
              });
            }
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
