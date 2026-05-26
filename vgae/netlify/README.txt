Netlify drag-and-drop deploy folder for the Mora visualization.

Upload this folder:
D:\The_Mora\vgae\netlify

This is the static version of viz_system:
- index.html, app.js, styles.css are the browser app.
- api/*.json files replace the original Python /api endpoints.
- api/chapter/*.json contains prebuilt chapter views.

Do not upload the whole D:\The_Mora\vgae directory to Netlify; it contains raw data,
model files, node_modules, local caches, and Python scripts that Netlify drag-and-drop
does not need.
