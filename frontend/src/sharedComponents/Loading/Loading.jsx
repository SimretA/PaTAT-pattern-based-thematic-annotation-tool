import * as React from "react";
import Backdrop from "@mui/material/Backdrop";
import Box from "@mui/material/Box";
import Modal from "@mui/material/Modal";
import Fade from "@mui/material/Fade";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";

import { useDispatch, useSelector } from "react-redux";
import {
  deleteSoftmatch,
  explainPattern,
  updatePatExp,
} from "../../actions/Dataslice";
import { CircularProgress, Divider } from "@mui/material";
import Chip from "@mui/material/Chip";

const style = {
  position: "absolute",
  top: "50%",
  left: "50%",
  transform: "translate(-50%, -50%)",
  width: 400,
  bgcolor: "background.paper",
  border: "1px solid #cccccc",
  boxShadow: 24,
  p: 4,
};

export default function CustomLoading() {
  const workspace = useSelector((state) => state.workspace);

  return (
    <Backdrop
      sx={{ color: "#fff", zIndex: (theme) => theme.zIndex.modal + 1 }}
      open={workspace.loading}
    >
      <CircularProgress
        color="inherit"
        sx={{
          ...style,
          flexShrink: 0,
          left: 0,
          right: 0,
          margin: "0 auto",
          opacity: 0.7,
          backgroundColor: workspace.color_code[workspace.selectedTheme],
        }}
      />
    </Backdrop>
  );
}
