import * as React from "react";
import Backdrop from "@mui/material/Backdrop";
import Box from "@mui/material/Box";
import Modal from "@mui/material/Modal";
import Fade from "@mui/material/Fade";
import Typography from "@mui/material/Typography";
import Stack from "@mui/material/Stack";

import { useDispatch, useSelector } from "react-redux";
import { deleteSoftmatch, explainPattern } from "../../actions/pattern_actions";
import {
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

const PatternViewComponent = (props) => {
  return (
    <Stack>
      <Typography
        variant="h6"
        component="h2"
        sx={{
          backgroundColor: "#FFFFFF",
          position: "sticky",
          top: "0px",
          zIndex: 10,
        }}
      >
        {props.value}
      </Typography>

      <Typography variant="overline">
        {!props.is_softmatch ? <>{props.explanation[0]}</> : <>Soft Match</>}{" "}
      </Typography>
      <Divider orientation="horizontal" flexItem />
      {props.is_softmatch && (
        <Stack spacing={2} mt={2}>
          {props.explanation.map((softmatch_value, index) => (
            <Chip
              onClick={(event) => {}}
              onDelete={() =>
                props.handleDelete(props.value, softmatch_value, props.value)
              }
              key={`soft_${softmatch_value}_${index}`}
              label={softmatch_value}
            />
          ))}
        </Stack>
      )}
    </Stack>
  );
};

export default function ExplainPattern({ open, setOpen, setRow, row }) {
  const handleDelete = (value, softmatch_value, pivot_word) => {
    dispatch(updatePatExp({ pattern: value, soft_match: softmatch_value }));
    // dispatch(deleteSoftmatch({ pattern: value, soft_match: softmatch_value }));

    dispatch(
      deleteSoftmatch({ pivot_word: pivot_word, similar_word: softmatch_value })
    );
  };

  const workspace = useSelector((state) => state.workspace);

  const dispatch = useDispatch();

  const handleExplain = (row) => {
    dispatch(explainPattern({ pattern: row["pattern"] }));
  };

  React.useEffect(() => {
    if (open) {
      handleExplain(row);
    }
  }, [open]);

  const handleClose = () => {
    setOpen(false);
    setRow(null);
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
    >
      <Fade in={open}>
        <Box
          sx={style}
          style={{
            minwidth: "40vw",
            maxWidth: "50vw",
            maxHeight: "80vh",
            minHeight: "40vh",
            overflow: "scroll",
          }}
        >
          <Typography id="transition-modal-title" variant="h6" component="h2">
            {row && row["pattern"]}
          </Typography>
          <Divider />
          <Stack
            divider={<Divider orientation="vertical" flexItem />}
            id="transition-modal-description"
            sx={{ mt: 2 }}
            direction="row"
            spacing={2}
          >
            {workspace.patternExp &&
              Object.keys(workspace.patternExp).map((value, index) => (
                <PatternViewComponent
                  handleDelete={handleDelete}
                  key={`exp_${index}_${value}`}
                  value={value}
                  is_softmatch={workspace.patternExp[value][0]}
                  explanation={workspace.patternExp[value][1]}
                />
              ))}

            {!workspace.patternExp && (
              <CircularProgress
                sx={{
                  ...style,
                  flexShrink: 0,
                  left: 0,
                  right: 0,
                  margin: "0 auto",
                  opacity: 0.7,
                  backgroundColor:
                    workspace.color_code[workspace.selectedTheme],
                }}
              />
            )}
          </Stack>
        </Box>
      </Fade>
    </Modal>
  );
}
