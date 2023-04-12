import * as React from "react";
import Typography from "@mui/material/Typography";
import { Stack } from "@mui/material";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import HighlightOffIcon from "@mui/icons-material/HighlightOff";
import IconButton from "@mui/material/IconButton";
import { useEffect } from "react";
import Highlighter from "react-highlight-words";
import { useSelector } from "react-redux";

import "./index.css";
import { ListItem } from "@material-ui/core";

export default function SentenceLight(props) {
  const workspace = useSelector((state) => state.workspace);
  const [labeled, setLabeled] = React.useState(
    workspace.element_to_label[props.element.id] &&
      workspace.element_to_label[props.element.id].includes(
        workspace.selectedTheme
      )
      ? 1
      : workspace.negative_element_to_label[props.element.id] &&
        workspace.negative_element_to_label[props.element.id].includes(
          workspace.selectedTheme
        )
      ? 0
      : null
  );
  const [toHighlight, setToHighlight] = React.useState([]);

  const handleClick = (id, label) => {
    setLabeled(label);
    props.handleBatchLabel(id, label);
  };

  useEffect(() => {
    if (props.highlight) {
      let to_highlight = [];
      props.highlight.forEach((element, index) => {
        if (props.highlightAsList) {
          let phrase = element[0] && element[0].join(" ");
          if (phrase) {
            to_highlight.push(phrase);
          }
        } else {
          to_highlight.push(element);
        }
      });

      setToHighlight(to_highlight);
    }
  }, []);

  const Highlight = ({ children, highlightIndex }) => (
    <span className="highlight" style={{ backgroundColor: props.color }}>
      {children}
    </span>
  );

  return (
    <>
      {props.show ? (
        <ListItem onClick={() => {}}>
          <Stack
            sx={{
              backgroundColor: "#eeeeee55",
              width: "100%",
              justifyContent: "space-between",
            }}
            direction={"row"}
            alignItems={"top"}
          >
            <Typography
              color="text.secondary"
              align="left"
              variant="body2"
              display="block"
              gutterBottom
            >
              {!props.highlight && <>{props.element.example}</>}
              {props.highlight && (
                <Highlighter
                  highlightTag={Highlight}
                  searchWords={toHighlight}
                  textToHighlight={props.element.example}
                />
              )}
            </Typography>

            {props.handleBatchLabel && (
              <Stack direction={"row"} alignItems={"top"}>
                <IconButton onClick={() => handleClick(props.element.id, 1)}>
                  {" "}
                  <CheckCircleOutlineIcon
                    color={labeled == 1 ? "success" : "disabled"}
                  />
                </IconButton>
                <IconButton onClick={() => handleClick(props.element.id, 0)}>
                  {" "}
                  <HighlightOffIcon
                    color={labeled == 0 ? "error" : "disabled"}
                  />
                </IconButton>
              </Stack>
            )}
          </Stack>
        </ListItem>
      ) : (
        <></>
      )}
    </>
  );
}
