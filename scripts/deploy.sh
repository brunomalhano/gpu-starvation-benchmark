#!/usr/bin/env bash
# deploy.sh — Full deployment for the PPD experiment on Azure Container Apps
#
# Prerequisites:
#   - Azure CLI installed and authenticated (az login)
#   - Subscription selected (az account set -s <subscription-id>)
#   - Docker is not required (uses az acr build)
#
# Usage:
#   chmod +x scripts/deploy.sh
#   ./scripts/deploy.sh
#
# Optional environment variables:
#   RG             — Resource Group (default: rg-trainning-models)
#   LOCATION       — Azure region (default: eastus)
#   DATASET_LOCAL  — Local dataset path to upload (default: ./_dataset_local)
#   ACR_NAME       — ACR name (auto-generated if omitted)
#   STORAGE_NAME   — Storage Account name (auto-generated if omitted)
#
# IMPORTANT: GPU workload profiles (NC24-A100) on Container Apps require:
#   1. A region with Container Apps GPU support (eastus, westus2, northeurope, etc.)
#   2. vCPU quota for the NC family in the subscription
#   Check quota with:
#     az quota list --scope /subscriptions/<sub-id>/providers/Microsoft.Compute/locations/<location> \
#       --query "[?contains(name.value,'NC')]" -o table

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────

RG="${RG:-rg-trainning-models}"
LOCATION="${LOCATION:-eastus}"
ENV_NAME="cae-ppd-experiment"
IMAGE_NAME="ppd-experiment"
IMAGE_TAG="v1"
DATASET_LOCAL="${DATASET_LOCAL:-./_dataset_local}"

# Generate a unique deterministic suffix from the RG name
UNIQUE_ID=$(printf '%s' "$RG" | shasum -a 256 | head -c 6)
ACR_NAME="${ACR_NAME:-acrppdexp${UNIQUE_ID}}"
STORAGE_NAME="${STORAGE_NAME:-stppdexp${UNIQUE_ID}}"
CONTAINER_IMAGE="${ACR_NAME}.azurecr.io/${IMAGE_NAME}:${IMAGE_TAG}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║   PPD Experiment — Azure Container Apps Deployment      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Resource Group:  $RG"
echo "║  Location:        $LOCATION"
echo "║  ACR:             $ACR_NAME"
echo "║  Storage:         $STORAGE_NAME"
echo "║  Image:           $CONTAINER_IMAGE"
echo "║  Dataset local:   $DATASET_LOCAL"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Resource Group ────────────────────────────────────────────

echo "[1/4] Creating resource group..."
az group create -n "$RG" -l "$LOCATION" -o none
echo "      ✓ Resource group ready"

# ── Step 2: Deploy Infrastructure (Bicep) ─────────────────────────────

echo "[2/4] Provisioning infrastructure (ACR, Storage, Container Apps Environment, Jobs)..."
echo "      This can take 5-10 minutes..."
az deployment group create \
  -g "$RG" \
  -f infra/main.bicep \
  --parameters \
    acrName="$ACR_NAME" \
    storageAccountName="$STORAGE_NAME" \
    environmentName="$ENV_NAME" \
    containerImage="$CONTAINER_IMAGE" \
  -o none
echo "      ✓ Infrastructure provisioned"

# ── Step 3: Build and Push Container Image ────────────────────────────

echo "[3/4] Building and pushing image to ACR..."
az acr build -r "$ACR_NAME" -t "${IMAGE_NAME}:${IMAGE_TAG}" . --no-logs
echo "      ✓ Image ${CONTAINER_IMAGE} published"

# ── Step 4: Upload Dataset ────────────────────────────────────────────

if [[ -d "$DATASET_LOCAL" ]]; then
    echo "[4/4] Uploading dataset to Azure Files share 'dataset'..."
    STORAGE_KEY=$(az storage account keys list \
      -g "$RG" -n "$STORAGE_NAME" \
      --query '[0].value' -o tsv)

    az storage file upload-batch \
      -d dataset \
      -s "$DATASET_LOCAL" \
      --account-name "$STORAGE_NAME" \
      --account-key "$STORAGE_KEY" \
      --no-progress \
      -o none
    echo "      ✓ Dataset uploaded"
else
    echo "[4/4] WARNING: Directory '$DATASET_LOCAL' not found."
    echo "      Upload it manually later:"
    echo "      az storage file upload-batch -d dataset -s /path/to/dataset \\"
    echo "        --account-name $STORAGE_NAME --account-key <key>"
fi

  # ── Summary ───────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Deployment Completed!                      ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║                                                         ║"
echo "║  Start CPU experiment:                                  ║"
echo "║    az containerapp job start \\                          ║"
echo "║      -n job-experiment-cpu -g $RG"
echo "║                                                         ║"
echo "║  Start GPU experiment:                                  ║"
echo "║    az containerapp job start \\                          ║"
echo "║      -n job-experiment-gpu -g $RG"
echo "║                                                         ║"
echo "║  Monitor executions:                                    ║"
echo "║    az containerapp job execution list \\                 ║"
echo "║      -n job-experiment-cpu -g $RG -o table"
echo "║    az containerapp job execution list \\                 ║"
echo "║      -n job-experiment-gpu -g $RG -o table"
echo "║                                                         ║"
echo "║  Download results:                                      ║"
echo "║    az storage file download-batch \\                     ║"
echo "║      -d ./results -s output \\                          ║"
echo "║      --account-name $STORAGE_NAME \\                    ║"
echo "║      --account-key \$(az storage account keys list \\    ║"
echo "║        -g $RG -n $STORAGE_NAME \\                       ║"
echo "║        --query '[0].value' -o tsv)                      ║"
echo "║                                                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
