FROM rocker/r-ver:4.3.2
WORKDIR /app

# Copy project (includes data + R files)
COPY ../ ./      

# Install R packages
RUN Rscript src/R_run/install_packages.R

# Run the R script
CMD ["Rscript", "src/R_run/model.R", "--data_dir", "src/data"]
