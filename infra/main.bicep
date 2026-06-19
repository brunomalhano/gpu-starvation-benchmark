// main.bicep
// Infrastructure for the PPD experiment on Azure Container Apps.
// Creates: ACR, Storage Account (Azure Files), Container Apps Environment
// with CPU and GPU workload profiles, and two Jobs (CPU and GPU).
//
// Usage:
//   az deployment group create -g rg-trainning-models -f infra/main.bicep \
//     --parameters acrName=<name> storageAccountName=<name> containerImage=<acr>.azurecr.io/ppd-experiment:v1

// ── Parameters ───────────────────────────────────────────────────────

@description('Azure region (defaults to the Resource Group region)')
param location string = resourceGroup().location

@description('Container Registry name (globally unique, alphanumeric only)')
param acrName string

@description('Storage Account name (globally unique, lowercase + numbers, 3-24 chars)')
param storageAccountName string

@description('Container Apps Environment name')
param environmentName string = 'cae-ppd-experiment'

@description('Full container image (e.g., myacr.azurecr.io/ppd-experiment:v1)')
param containerImage string

@description('Dedicated CPU workload profile type (D4=4vCPU/16GiB, D8=8vCPU/32GiB)')
param cpuProfileType string = 'D8'

@description('GPU workload profile type (NC24-A100, NC48-A100). Check regional availability.')
param gpuProfileType string = 'NC24-A100'

@description('Number of DataLoader workers (0 = main thread only)')
param numWorkers int = 0

// ── Variables ────────────────────────────────────────────────────────

var identityName = 'id-ppd-experiment'
var logAnalyticsName = 'log-ppd-experiment'
var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'

// ── Log Analytics (required for Container Apps) ──────────────────────

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── Managed Identity (ACR pull without credentials) ───────────────────

resource managedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

// ── Container Registry ───────────────────────────────────────────────

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: false
  }
}

// ── Role Assignment: AcrPull for the Managed Identity ─────────────────

resource acrPullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(acr.id, managedIdentity.id, acrPullRoleId)
  scope: acr
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      acrPullRoleId
    )
    principalId: managedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// ── Storage Account + Azure Files Shares ─────────────────────────────

resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowSharedKeyAccess: true
  }
}

resource fileServices 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storageAccount
  name: 'default'
}

resource datasetShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileServices
  name: 'dataset'
  properties: {
    shareQuota: 10
  }
}

resource outputShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileServices
  name: 'output'
  properties: {
    shareQuota: 1
  }
}

// ── Container Apps Environment ───────────────────────────────────────

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
      {
        name: 'cpu-dedicated'
        workloadProfileType: cpuProfileType
        minimumCount: 0
        maximumCount: 1
      }
      {
        name: 'gpu-dedicated'
        workloadProfileType: gpuProfileType
        minimumCount: 0
        maximumCount: 1
      }
    ]
  }
}

// ── Environment Storages (bind Azure Files to the Environment) ───────

resource envStorageDataset 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: environment
  name: 'dataset-storage'
  properties: {
    azureFile: {
      accountName: storageAccount.name
      accountKey: storageAccount.listKeys().keys[0].value
      shareName: datasetShare.name
      accessMode: 'ReadOnly'
    }
  }
}

resource envStorageOutput 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: environment
  name: 'output-storage'
  properties: {
    azureFile: {
      accountName: storageAccount.name
      accountKey: storageAccount.listKeys().keys[0].value
      shareName: outputShare.name
      accessMode: 'ReadWrite'
    }
  }
}

// ── Job CPU ──────────────────────────────────────────────────────────

resource jobCpu 'Microsoft.App/jobs@2024-03-01' = {
  name: 'job-experiment-cpu'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    environmentId: environment.id
    workloadProfileName: 'cpu-dedicated'
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 7200
      replicaRetryLimit: 0
      registries: [
        {
          server: '${acr.name}.azurecr.io'
          identity: managedIdentity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'experiment'
          image: containerImage
          resources: {
            cpu: json('8')
            memory: '16Gi'
          }
          env: [
            { name: 'DATASET_PATH', value: '/mnt/dataset' }
            { name: 'OUTPUT_PATH', value: '/mnt/output' }
            { name: 'FORCE_CPU', value: '1' }
            { name: 'NUM_WORKERS', value: '${numWorkers}' }
          ]
          volumeMounts: [
            { mountPath: '/mnt/dataset', volumeName: 'dataset-vol' }
            { mountPath: '/mnt/output', volumeName: 'output-vol' }
          ]
        }
      ]
      volumes: [
        {
          name: 'dataset-vol'
          storageName: envStorageDataset.name
          storageType: 'AzureFile'
        }
        {
          name: 'output-vol'
          storageName: envStorageOutput.name
          storageType: 'AzureFile'
        }
      ]
    }
  }
}

// ── Job GPU ──────────────────────────────────────────────────────────

resource jobGpu 'Microsoft.App/jobs@2024-03-01' = {
  name: 'job-experiment-gpu'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${managedIdentity.id}': {}
    }
  }
  properties: {
    environmentId: environment.id
    workloadProfileName: 'gpu-dedicated'
    configuration: {
      triggerType: 'Manual'
      replicaTimeout: 7200
      replicaRetryLimit: 0
      registries: [
        {
          server: '${acr.name}.azurecr.io'
          identity: managedIdentity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'experiment'
          image: containerImage
          resources: {
            cpu: json('12')
            memory: '32Gi'
          }
          env: [
            { name: 'DATASET_PATH', value: '/mnt/dataset' }
            { name: 'OUTPUT_PATH', value: '/mnt/output' }
            { name: 'NUM_WORKERS', value: '${numWorkers}' }
          ]
          volumeMounts: [
            { mountPath: '/mnt/dataset', volumeName: 'dataset-vol' }
            { mountPath: '/mnt/output', volumeName: 'output-vol' }
          ]
        }
      ]
      volumes: [
        {
          name: 'dataset-vol'
          storageName: envStorageDataset.name
          storageType: 'AzureFile'
        }
        {
          name: 'output-vol'
          storageName: envStorageOutput.name
          storageType: 'AzureFile'
        }
      ]
    }
  }
}

// ── Outputs ──────────────────────────────────────────────────────────

output acrLoginServer string = acr.properties.loginServer
output environmentName string = environment.name
output storageAccountName string = storageAccount.name
output jobCpuName string = jobCpu.name
output jobGpuName string = jobGpu.name
