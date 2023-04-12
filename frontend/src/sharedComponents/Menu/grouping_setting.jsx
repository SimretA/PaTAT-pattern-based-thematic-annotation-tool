import * as React from "react";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemText from "@mui/material/ListItemText";
import MenuItem from "@mui/material/MenuItem";
import Menu from "@mui/material/Menu";
import AutoFixHighIcon from "@mui/icons-material/AutoFixHigh";
import ListItemIcon from "@mui/material/ListItemIcon";
import { useDispatch, useSelector } from "react-redux";
import { fetchGroupedDataset } from "../../actions/dataset_actions";
import {
  changeGroupingSetting,
  changeSetting,
} from "../../actions/Dataslice";
import Radio from "@mui/material/Radio";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import FormControl from "@mui/material/FormControl";
import FormLabel from "@mui/material/FormLabel";
import { Typography } from "@material-ui/core";
import Grid from "@mui/material/Grid";
import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";
const styles = (theme) => ({
  checked: {},
});

export default function GroupingSettings(props) {
  const workspace = useSelector((state) => state.workspace);

  const dispatch = useDispatch();

  const options = workspace.groupingSettings;

  const [value, setValue] = React.useState("");

  const [anchorEl, setAnchorEl] = React.useState(null);
  const open = Boolean(anchorEl);
  const handleClickListItem = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuItemClick = (event, index) => {
    dispatch(changeGroupingSetting({ selectedSetting: event.target.value }));

    dispatch(fetchGroupedDataset({ selectedSetting: event.target.value }));

    setAnchorEl(null);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <FormControl>
      <InputLabel id="demo-simple-select-label">{"Group By"}</InputLabel>
      <Select
        labelId="demo-simple-select-label"
        id="demo-simple-select"
        value={workspace.selectedGroupSetting}
        label="Group By"
        onChange={(event) => handleMenuItemClick(event, event.target.value)}
        size="small"
      >
        {Object.keys(options).map((key, index) => (
          <MenuItem
            key={`grouping_${key}`}
            value={key}
            disabled={
              (key == 1 &&
                Object.keys(workspace.selectedPatterns).length < 1) ||
              (key == 2 && Object.keys(workspace.element_to_label).length < 1)
            }
          >
            {options[key]}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}
