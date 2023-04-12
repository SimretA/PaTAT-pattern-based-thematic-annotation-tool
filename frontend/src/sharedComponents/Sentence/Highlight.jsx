import * as React from "react";
import {
  CloseOutlined,
  DeleteForeverRounded,
  PhonelinkEraseOutlined,
} from "@mui/icons-material";
import { useDispatch, useSelector } from "react-redux";
import { lighten } from "@material-ui/core";
import IconButton from "@mui/material/IconButton";
import "./index.css";
import CancelIcon from "@mui/icons-material/Cancel";
import { labelPhrase } from "../../actions/annotation_actions";

export default function Highlight(props) {
  const [anchorEl, setAnchorEl] = React.useState(null);
  const [patterns, setPatterns] = React.useState([]);
  const [showDelete, setShowDelete] = React.useState(false);

  const workspace = useSelector((state) => state.workspace);
  const dispatch = useDispatch();

  const getMatchingPatterns = () => {
    let pats = [];
    for (let index = props.start; index < props.end; index++) {
      pats = [...pats, ...props.matchedWith[index]];
    }
    pats = [...new Set(pats)];

    return pats;
  };
  const getChild = (patterns) => {
    const pats = getMatchingPatterns();

    return (
      <>
        {pats.map((pattern, index) => (
          <>
            <em key={`pat_${index}`}>{pattern}</em>:{" "}
            {pattern == "USER_DEFINED"
              ? `-`
              : `${parseFloat(workspace.selectedPatterns[pattern]).toFixed(2)}`}
            <br />
          </>
        ))}
      </>
    );
  };

  const handlePopoverOpen = (event) => {
    if (props.matched) {
      setShowDelete(true);

      getMatchingPatterns();

      props.setPopoverAnchor(event.currentTarget);
      props.setPopoverContent(getChild(patterns));

    }
  };

  const handlePopoverClose = () => {
    if (props.matched) {
      setShowDelete(false);
      props.setPopoverAnchor(null);
      props.setPopoverContent(null);
    }
  };

  const handleClick = () => {
    if (props.deleteMatched) {
      props.deleteMatched(
        props.word,
        0,
        props.start,
        props.end,
        getMatchingPatterns()
      );
    }
  };

  const get_highlight_coolor = (patterns) => {
    let sum = 0;
    if (workspace.selectedPatterns) {
      patterns.forEach((pattern) => {
        sum += workspace.selectedPatterns[pattern];
      });
      return sum;
    } else {
      return null;
    }
  };

  const open = Boolean(anchorEl);
  return (
    <span
      onMouseEnter={handlePopoverOpen}
      onMouseLeave={handlePopoverClose}
      style={{
        userSelect: "text",
        ...(props.matched &&
          (workspace.selectedPatterns[props.patterns[0]] > 0 ||
            props.patterns[0] == "USER_DEFINED") && {
            backgroundColor: workspace.color_code[workspace.selectedTheme]
              ? `${lighten(workspace.color_code[workspace.selectedTheme], 0.5)}`
              : `${lighten("#ececec", 0.5)}`,
          }),
      }}
      className={
        props.matched
          ? workspace.selectedPatterns[props.patterns[0]] < 0
            ? "highlight_red"
            : "highlight"
          : "non-highlight"
      }
    >
      {`${props.word.split(" ").slice(0, -1).join(" ")} `}
      {props.matched && showDelete && (
        <IconButton
          style={{
            position: "relative",
            top: "-15px",
            right: "-10px",
            width: "25px",
            height: "25px",
          }}
          size="small"
          onClick={() => {
            handlePopoverClose();
            handleClick(props.word, 0);
            const phrase = `${props.word.split(" ").slice(0, -1).join(" ")} `;
            if (phrase.trim() != "") {
              dispatch(
                labelPhrase({
                  phrase: phrase,
                  label: workspace.selectedTheme,
                  id: props.elementId,
                  positive: 0,
                })
              ).then(() => {
                window.getSelection().empty();
              });
            }
          }}
        >
          <CancelIcon
            color="error"
            style={{
              opacity: 0.3,
              width: "25px",
              height: "25px",
            }}
          />
        </IconButton>
      )}
    </span>
  );
}
