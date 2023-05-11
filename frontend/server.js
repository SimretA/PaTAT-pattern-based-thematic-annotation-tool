const express = require("express");
const app = express();
const path = require("path");
app.use(express.static(path.join(__dirname, "build")));
app.get("/", function (req, res) {
  res.sendFile(path.join(__dirname, "build", "index.html"));
});
app.listen(3000, () => {
  console.log("server is runnig on port 3000");
  console.log("Open your browser and head to 'http://localhost:3000/'");
});
