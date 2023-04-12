import React from "react";
import { Button, LinearProgress } from "@mui/material";
import { makeStyles, withStyles } from "@material-ui/core/styles";

const useStyles = makeStyles((theme) => ({
  root: {
    marginLeft: theme.spacing(2),
    marginRight: theme.spacing(2),
    flexGrow: 1,
    alignItems: "center",
    justifyContent: "center",
  },
  button: {
    margin: `${theme.spacing(1)} auto`,
  },
  progress: {
    height: "50px",
  },
  colorPrimary: {
    backgroundColor: "green", //props => props.color,
  },

  barColorPrimary: {
    backgroundColor: "red",
  },
  bar: (props) => ({
    backgroundColor: props.color,
  }),
}));

const BorderLinearProgress = withStyles((theme) => ({
  root: {
    height: 10,
    borderRadius: 5,
  },
  colorPrimary: {
    backgroundColor: "#e2e8e5 !important",
  },
  bar: (props) => ({
    borderRadius: 5,
    backgroundColor: `${props.bgColor} !important`,
  }),
}))(LinearProgress);

export default function ProgressButton(props) {
  const classes = useStyles();

  return (
    <div className={classes.root}>
      <Button
        onClick={() => {
          props.retrain();
        }}
        size={"small"}
        className={classes.button}
        sx={{ backgroundColor: props.color }}
        variant="contained"
      >
        <div style={{ fontWeight: "bold" }}>
          {/* <BorderLinearProgress  bgColor={props.color} sx={{minHeight:"20px"}}  variant="determinate" value={props.value} /> */}
          Retrain in{" "}
          {props.userAnnotationTracker < props.annotationPerRetrain
            ? `${props.annotationPerRetrain - props.userAnnotationTracker}`
            : "0"}
        </div>
      </Button>
    </div>
  );
}
