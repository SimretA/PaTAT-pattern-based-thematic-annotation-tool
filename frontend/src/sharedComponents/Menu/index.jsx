import * as React from "react";
import MenuItem from "@mui/material/MenuItem";
import { useDispatch, useSelector } from "react-redux";
import { changeSetting } from "../../actions/Dataslice";
import FormControl from "@mui/material/FormControl";
import { fetchGroupedDataset } from "../../actions/dataset_actions";

import InputLabel from "@mui/material/InputLabel";
import Select from "@mui/material/Select";

export default function Settings(props) {
  const workspace = useSelector((state) => state.workspace);

  const dispatch = useDispatch();

  const options = workspace.orderSetting;

  const [anchorEl, setAnchorEl] = React.useState(null);

  const handleMenuItemClick = (event, index) => {
    dispatch(changeSetting({ selectedSetting: index }));
    if (index == 0 && workspace.selectedGroupSetting == 0)
      dispatch(fetchGroupedDataset({ selectedSetting: 0 }));
    setAnchorEl(null);
  };

  return (
    <FormControl>
      <InputLabel id="demo-simple-select-label">{"Rank By"}</InputLabel>
      <Select
        labelId="demo-simple-select-label"
        id="demo-simple-select"
        value={workspace.selectedSetting}
        label="Rank By"
        onChange={(event) => handleMenuItemClick(event, event.target.value)}
        size="small"
      >
        {Object.keys(options).map((key, index) => (
          <MenuItem key={`ordering_${key}`} value={key}>
            {options[key]}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}
