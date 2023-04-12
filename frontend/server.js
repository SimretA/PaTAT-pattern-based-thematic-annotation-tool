const express = require("express");
const app = express();
const path = require("path");
app.use(express.static(path.join(__dirname, "build")));
app.get("/", function (req, res) {
  res.sendFile(path.join(__dirname, "build", "index.html"));
});
app.listen(8282, () => {
  console.log("server is r  unnig on port 8282");
  console.log("Open your browser and hit url 'http://34.68.65.252:8282/'");
});
