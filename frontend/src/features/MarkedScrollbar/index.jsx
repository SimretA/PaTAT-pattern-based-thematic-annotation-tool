import React from "react";
import { Stack, Paper } from "@mui/material";
import { darken, lighten } from "@material-ui/core";

import { useSelector } from "react-redux";

export default function Scroller(props) {
  const workspace = useSelector((state) => state.workspace);

  const [ids, setIds] = React.useState([]);

  React.useEffect(() => {
    if (workspace.groups && workspace.groups.length == 1) {
      setIds(...workspace.groups);
    } else if (workspace.groups && workspace.groups.length > 1) {
      let new_ids = [];
      workspace.groups.forEach((group) => {
        new_ids = new_ids.concat(group);
      });
      setIds(new_ids);
    }
  }, [workspace.groups]);

  return (
    <Stack
      mt={"145px"}
      direction="column"
      sx={{
        ...(!props.show && { visibility: "hidden" }),
        backgroundColor: "none",
        height: "84vh",
        width: "10px",
        position: "absolute",
        left: "0vw",
        zIndex: 10,
      }}
    >
      {ids &&
        ids.length > 0 &&
        ids.map((elementId) => {
          let data = workspace.elements[elementId];

          return (
            <Paper
              sx={{
                backgroundColor:
                  data.score &&
                  data.score > 0.5 &&
                  workspace.color_code[workspace.selectedTheme]
                    ? `${darken(
                        workspace.color_code[workspace.selectedTheme],
                        (data.score - 0.5) / 0.5
                      )}`
                    : data.score && data.score < 0.5
                    ? `${lighten("#fc0b22", 1 - data.score)}`
                    : "none",
                height: `${(window.innerHeight - 140) / ids.length}px`,
                width: "15px",
                borderRadius: "0px",
              }}
              key={`scroller_${data.id}`}
            />
          );
        })}

      <Paper
        sx={{
          height: `${(window.innerHeight - 140) / ids.length}px`,
          width: "15px",
          backgroundColor: "#b3bcff",
          borderRadius: "0px",
          position: "absolute",
          top: `${props.scrollPosition * (window.innerHeight - 140)}px`,
          border: "none",
          opacity: "1",
        }}
      />
    </Stack>
  );
}
