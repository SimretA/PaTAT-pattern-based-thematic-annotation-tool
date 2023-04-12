import * as React from "react";

import Typography from "@mui/material/Typography";
import { Divider, Button, IconButton } from "@mui/material";
import { Stack, Chip } from "@mui/material";
import { useDispatch, useSelector } from "react-redux";
import DriveFileRenameOutlineIcon from "@mui/icons-material/DriveFileRenameOutline";

import SplitThemeModal from "./SplitThemeModal";
export default function Summary(props) {
  const workspace = useSelector((state) => state.workspace);
  const dispatch = useDispatch();

  const [action, setAction] = React.useState(null);
  const [actionType, setActionType] = React.useState(null);
  const [open, setOpen] = React.useState(false);

  return (
    <>
      <SplitThemeModal
        open={open}
        setOpen={setOpen}
        action={action}
        actionType={actionType}
        retrain={props.retrain}
        theme={workspace.selectedTheme}
      />
      <Stack direction={"row"} spacing={0}>
        <Typography variant="h5">
          Patterns for{" "}
          <Chip
            key={`menuitem_new_theme_light_summary`}
            label={""}
            color={"primary"}
            sx={{
              backgroundColor: props.color,
              width: 20,
              height: 20,
              marginRight: 1,
              mr: 1,
              my: 0.5,
            }}
            size="small"
          />
          {props.selectedTheme}
        </Typography>
        <Button
          onClick={(event) => {
            // setAnchorEl(event.target)
            setActionType("rename");
            // setAction(handleMerge);
            setOpen(true);
          }}
        >
          Rename
        </Button>
        <Divider orientation="vertical" />
        <Button
          onClick={(event) => {
            // setAnchorEl(event.target)
            setActionType("merge");
            // setAction(handleMerge);
            setOpen(true);
          }}
        >
          Merge
        </Button>
        <Divider orientation="vertical" />
        <Button
          onClick={(event) => {
            setActionType("split");
            setOpen(true);
          }}
        >
          Split
        </Button>
        <Divider orientation="vertical" />
        <Button
          onClick={(event) => {
            setActionType("delete");
            setOpen(true);
          }}
        >
          Delete
        </Button>
      </Stack>
    </>
  );
}
