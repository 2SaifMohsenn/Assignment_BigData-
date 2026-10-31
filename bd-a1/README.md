# CSCI461 Assignment 1 — SMS Spam Text Analysis

This project uses the **SMS Spam Collection**, an approved UCI dataset, to analyze ham and spam messages. The supplied lab notebooks informed the workflow: preprocessing and K-Means (Lab 4), exploratory analysis (Lab 3), and tokenization, TF-IDF, and spam filtering (Lab 5). The supplied student-record JSON is not used because it is not one of the approved assignment datasets.

## Project layout

```text
bd-a1/
├── Dockerfile
├── requirements.txt
├── README.md
├── analysis.py
├── data/
│   └── dataset.csv
├── notebook/
│   └── analysis.ipynb
├── output/
│   ├── cleaned_data.csv
│   ├── model_metrics.txt
│   ├── cluster_summary.txt
│   └── figures/
└── screenshots/
```

The notebook invokes the shared `analysis.py` implementation so the exact same analysis runs inside Jupyter and can be verified without duplicating code. It reads only `data/dataset.csv` and writes generated files to `output/`.

## Dataset

The UCI SMS Spam Collection contains 5,574 labeled SMS messages. Each row represents one message, with `label` (`ham` or `spam`) and the original message in `text`.

- Dataset page: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>
- Official download: <https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip>
- Citation: Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). *Contributions to the study of SMS spam filtering: new collection and results*. Proceedings of the 2011 ACM Symposium on Document Engineering.

The downloaded archive is converted to the required comma-separated `data/dataset.csv` before the image is built. To reproduce the conversion on Windows PowerShell if needed:

```powershell
Invoke-WebRequest "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" -OutFile "data\sms-spam-collection.zip"
Expand-Archive "data\sms-spam-collection.zip" -DestinationPath "data\uci_sms" -Force
node -e "const fs=require('fs');const src=fs.readFileSync('data/uci_sms/SMSSpamCollection','utf8').trimEnd().split(/\r?\n/);const q=s=>'\"'+s.replace(/\"/g,'\"\"')+'\"';const rows=['label,text'];for(const line of src){const i=line.indexOf('\t');if(i<0)throw Error('Malformed UCI row');rows.push(q(line.slice(0,i))+','+q(line.slice(i+1)));}fs.writeFileSync('data/dataset.csv',rows.join('\n')+'\n','utf8');"
```

## Build and run with Docker Desktop (PowerShell)

Run these commands from the project directory. Docker Desktop must be running.

```powershell
docker build -t bd-a1-sms:latest .
docker run --name bd-a1-sms -p 8888:8888 -v "${PWD}\output:/home/jovyan/work/bd-a1/output" bd-a1-sms:latest
```

Open the JupyterLab URL printed by `docker logs bd-a1-sms` (usually `http://localhost:8888/lab`) and run `notebook/analysis.ipynb` with **Kernel → Restart Kernel and Run All Cells**. The bind mount makes generated reports and figures available in the host's `output\` folder, independently of the container's lifecycle.

`docker build` creates an **image**: an immutable template containing the base environment, dependencies, dataset, and project files. `docker run` creates a **container**: a running instance of that image. In `-p 8888:8888`, the first port is the **host** port and the second is the **container** port; requests to host port 8888 are forwarded to JupyterLab's port 8888 inside the container.

To run and execute the notebook in the container in one reproducibility check (writes outputs to the same host bind mount):

```powershell
docker run --rm -v "${PWD}\output:/home/jovyan/work/bd-a1/output" bd-a1-sms:latest jupyter nbconvert --to notebook --execute --inplace notebook/analysis.ipynb
```

The command above executes the notebook but does not launch the Jupyter web server. For interactive work use the `docker run` command in the previous section.

### Container lifecycle evidence

With the interactive container running, capture evidence and save screenshots in `screenshots\`:

```powershell
docker images
docker ps
docker ps -a
docker logs bd-a1-sms
docker inspect bd-a1-sms
```

The `screenshots\jupyter-notebook.png` file is a genuine capture of the assignment notebook open in the running JupyterLab container. `screenshots\docker-hub-repository.png` is a genuine capture of the published Docker Hub repository page. Still add genuine captures of the successful image build, `docker images`, and the running `docker ps` entry from the team's Docker environment, plus the teammate verification screenshot, before submission; do not use fabricated evidence.

## Analysis and expected outputs

The notebook reports dataset structure and cleaning decisions; demonstrates `loc`, `iloc` with `numpy.r_`, three sampling approaches, feature creation, text normalization and word frequencies; plots numeric-feature correlations and group comparisons; compares Multinomial Naive Bayes with Logistic Regression using stratified train/test data and TF-IDF; and evaluates K-Means for candidate values of *k* using sampled silhouette scores. It writes six figures, a cleaned CSV, model metrics, a cluster summary, and six evidence-based written insights beneath `output/`.

The K-Means length-quartile sampling example is explicitly exploratory: length-defined clusters are not naturally independent groups and therefore may not represent a production sampling design. Likewise, correlation is descriptive and does not establish causation; classification scores can be affected by class imbalance and by the small, historically collected SMS corpus.

## Docker Hub distribution

Docker Hub image: [`1120055/bd-a1-sms:latest`](https://hub.docker.com/r/1120055/bd-a1-sms). The image was pushed to Docker Hub and then pulled successfully from the registry. Published manifest digest: `sha256:88f6316937ec868db3c25d525d86885008ec2ff0706ccaeffd2aeecfde65ba71`. The commands below reproduce the build and publication:

```powershell
$Image = "1120055/bd-a1-sms:latest"
docker build -t $Image .
docker login
docker push $Image
```

Repository: `1120055/bd-a1-sms`; tag: `latest`. A registry pull and notebook execution from the published image both succeeded in this environment. **Assignment handoff still required:** a different team member must independently pull and run the image, execute the notebook, and return a genuine verification screenshot:

```powershell
docker pull 1120055/bd-a1-sms:latest
docker run --name bd-a1-sms-pulled -p 8888:8888 -v "${PWD}\output:/home/jovyan/work/bd-a1/output" 1120055/bd-a1-sms:latest
```

Confirm that the pulled image starts JupyterLab and that running the notebook creates the expected host-side outputs. Add a genuine verification screenshot to `screenshots\`.

## Submission ZIP

After completing the remaining Docker Hub, teammate-verification, and Docker lifecycle screenshot steps, recreate the ZIP from the parent directory:

```powershell
Compress-Archive -Path .\bd-a1 -DestinationPath .\bd-a1.zip -Force
```

Include the dataset unless its redistribution terms prohibit it; if omitted, include the official download and conversion instructions above.
