import * as React from "react";
import Popover from "@mui/material/Popover";
import { Button, Stack, Typography } from "@mui/material";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import HighlightOffIcon from "@mui/icons-material/HighlightOff";
import IconButton from "@mui/material/IconButton";
import { useDispatch, useSelector } from "react-redux";

export default function CustomPopover(props) {
  const [anchorEl, setAnchorEl] = React.useState(null);

  const workspace = useSelector((state) => state.workspace);

  const { x, y } = props;

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  React.useEffect(() => {
    if (anchorEl != null) {
    }
  }, [anchorEl]);
  const open = Boolean(anchorEl);
  const id = open ? "simple-popover" : undefined;

  const phraseLabeling = (id, label) => {
    props.handlePhraseLabeling(label);
  };

  return (
    <Popover
      id={id}
      open={props.open}
      anchorPosition={{ left: x, top: y }}
      anchorReference={"anchorPosition"}
      onClose={props.handleClose}
      anchorOrigin={{
        vertical: "bottom",
        horizontal: "left",
      }}
      style={{
        borderRadius: "10px",
      }}
    >
      {workspace.selectedTheme ? (
        <Stack
          sx={{
            padding: "5px",
            border: `solid 3px ${
              workspace.color_code[workspace.selectedTheme]
            }`,
            backgroundColor: "#FFFFFF11",
            borderRadius: "10px",
          }}
          direction={"row"}
          spacing={5}
        >
          <Button
            backgroundColor={workspace.color_code[workspace.selectedTheme]}
            size={"small"}
            onClick={() => phraseLabeling("hello", 1)}
            variant={"outlined"}
            startIcon={<CheckCircleOutlineIcon color={"success"} />}
          >
            {" "}
            {workspace.selectedTheme}
          </Button>
          {/* <Button
            onClick={() => handleClose()}
            variant={"outlined"}
            color={"info"}
            size={"small"}
          >
            Cancel
          </Button> */}
        </Stack>
      ) : (
        <Typography>
          Please select theme for phrase level annotationß
        </Typography>
      )}
    </Popover>
  );
}
