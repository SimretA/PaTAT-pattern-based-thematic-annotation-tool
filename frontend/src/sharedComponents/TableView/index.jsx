import * as React from "react";
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";

import TableRow from "@mui/material/TableRow";
import { IconButton, LinearProgress, Tooltip } from "@material-ui/core";
import DeleteIcon from "@mui/icons-material/Delete";
import FilterAltIcon from "@mui/icons-material/FilterAlt";
import { useDispatch, useSelector } from "react-redux";
import InfoIcon from "@mui/icons-material/Info";
 import PushPinIcon from "@mui/icons-material/PushPin";
 import { deletePattern, pinPattern, fetchCombinedPatterns } from "../../actions/pattern_actions.jsx";
import {
  updatePatterns,
} from "../../actions/Dataslice.jsx";

export default function TabelView(props) {
  const [locked, setLocked] = React.useState(false);

  const workspace = useSelector((state) => state.workspace);

  const dispatch = useDispatch();

  const refresh = () => {
    dispatch(fetchCombinedPatterns()).then((response) => {});
  };
  const handleVisiblityLock = (row) => {
    if (!locked && props.setHovering) {
      props.setHovering(row["pattern"]);
      setLocked(true);
    } else if (locked && props.setHovering) {
      if (props.hovering == row["pattern"]) {
        setLocked(false);
        props.setHovering(null);
      } else {
        props.setHovering(row["pattern"]);
      }
    }
  };

  const handleDeletePattern = (row) => {
    dispatch(updatePatterns({ pattern: row["pattern"], status: 0 }));
    dispatch(
      deletePattern({
        theme: workspace.selectedTheme,
        pattern: row["pattern"],
      })
    ).then((response) => {
      refresh();
    });
  };

  const handlePinPattern = (row) => {
    dispatch(updatePatterns({ pattern: row["pattern"], status: 1 }));
    dispatch(
      pinPattern({
        theme: workspace.selectedTheme,
        pattern: row["pattern"],
      })
    ).then((response) => {
      refresh();
    });
  };

  const handleExplain = (event, row) => {
    props.handelOpenModal(row);
  };

  return (
    <Paper
      sx={{
        width: "100%",
        paddingBottom: "20px",
        overflow: "hidden",
        marginBottom: "5px",
      }}
    >
      {props.loading && <LinearProgress width={"100%"} />}
      <>
        <TableContainer>
          <Table stickyHeader aria-label="sticky table">
            <TableHead>
              <TableRow>
                {props.columns.map((column) => (
                  <TableCell
                    key={column.id}
                    align={column.align}
                    style={{ minWidth: column.minWidth }}
                    sx={{ backgroundColor: "#000000", color: "#FFFFFF" }}
                  >
                    {column.label}
                  </TableCell>
                ))}
                <TableCell
                  sx={{ backgroundColor: "#000000", color: "#FFFFFF" }}
                ></TableCell>
              </TableRow>
            </TableHead>

            <TableBody>
              {props.data &&
                props.data
                  // .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                  .map((row, index) => {
                    return (
                      <TableRow
                        hover
                        role="checkbox"
                        tabIndex={-1}
                        key={row.code}
                        sx={{
                          ...(props.hovering == row["pattern"] && {
                            backgroundColor: "#f5f5f5",
                          }),
                        }}
                      >
                        {props.columns.map((column, index) => {
                          const value = row[column.id];
                          return (
                            <TableCell
                              sx={{ ...(index == 0 && { fontWeight: 700 }) }}
                              key={column.id}
                              align={column.align}
                            >
                              {index == 0 && (
                                <IconButton
                                  onClick={(event) => handleExplain(event, row)}
                                >
                                  <InfoIcon />
                                </IconButton>
                              )}
                              {typeof value === "number"
                                ? parseFloat(value).toFixed(2)
                                : value}
                            </TableCell>
                          );
                        })}
                        {!workspace.loadingPatterns &&
                          !workspace.loadingCombinedPatterns && (
                            <>
                              <TableCell key={"filter"}>
                                {props.index == 0 && (
                                  <Tooltip
                                    title={
                                      props.hovering == row["pattern"]
                                        ? "Turn-off filter"
                                        : "Filter with pattern"
                                    }
                                  >
                                    <IconButton
                                      onClick={() => handleVisiblityLock(row)}
                                    >
                                      <FilterAltIcon
                                        color={
                                          props.hovering == row["pattern"]
                                            ? "primary"
                                            : "disabled"
                                        }
                                      />
                                    </IconButton>
                                  </Tooltip>
                                )}
                                {row["status"] == 1 ? (
                                  <IconButton>
                                    <DeleteIcon
                                      onClick={() => handleDeletePattern(row)}
                                    />
                                  </IconButton>
                                ) : (
                                  <IconButton>
                                    <PushPinIcon
                                      onClick={() => handlePinPattern(row)}
                                    />
                                  </IconButton>
                                )}
                              </TableCell>
                            </>
                          )}
                      </TableRow>
                    );
                  })}
            </TableBody>
          </Table>
        </TableContainer>
      </>
    </Paper>
  );
}
