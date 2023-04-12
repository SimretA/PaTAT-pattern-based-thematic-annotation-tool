import logo from '../assets/logo.svg';
import '../styles/App.css';
import React, { useState } from 'react';
import ReactDOM from 'react-dom';
import { Routes , Route } from 'react-router-dom';

import modules from '../routes';
import { MuiThemeProvider } from '@material-ui/core';
import { createTheme } from '@material-ui/core';
import { CssBaseline } from '@mui/material';
// import { Provider } from 'react-redux';


function App() {

  const THEME = createTheme({
    typography: {
     "fontFamily": `"Roboto", "Nunito", "Helvetica", "Arial", sans-serif`,
     "fontSize": 14,
     "fontWeightLight": 300,
     "fontWeightRegular": 400,
     "fontWeightMedium": 500
    },
    palette: {
      mode: darkmode?'dark':'light',
      background: {
        default: 'red',
      },
    },
    components: {
      // Name of the component
      MuiCard: {
        styleOverrides: {
          // Name of the slot
          root: {
            // Some CSS
            border:"none"
          },
        },
      },
    },
 });

 const [darkmode, setDarkMode] = useState(true)

  return (
      <div className="App">
        <MuiThemeProvider theme={THEME}>
          <CssBaseline/>
          <Routes>
            {modules.map(module => (
              <Route {...module.routeProps} key={module.name} />
            ))}
          </Routes>
        </MuiThemeProvider>
      </div>
  );
}

export default App;
