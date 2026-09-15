$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$assetDirectory = Join-Path $projectRoot "models\realworld_v2\upstream"
New-Item -ItemType Directory -Force -Path $assetDirectory | Out-Null

function Get-VerifiedAsset {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Target,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )
    if (Test-Path -LiteralPath $Target) {
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Target).Hash.ToLowerInvariant()
        if ($actual -eq $ExpectedSha256) {
            Write-Host "Verified existing asset: $Target"
            return
        }
        throw "Existing asset has the wrong SHA-256: $Target. Remove it manually before retrying."
    }

    Invoke-WebRequest -Uri $Url -OutFile $Target
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Target).Hash.ToLowerInvariant()
    if ($actual -ne $ExpectedSha256) {
        throw "Downloaded asset failed SHA-256 verification: $Target"
    }
    Write-Host "Downloaded and verified: $Target"
}

$clipHash = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
Get-VerifiedAsset `
    -Url "https://openaipublic.azureedge.net/clip/models/$clipHash/ViT-L-14.pt" `
    -Target (Join-Path $assetDirectory "ViT-L-14.pt") `
    -ExpectedSha256 $clipHash

$communityRevision = "6076002bf0d9dd37537f965ee2f06f826c333b61"
$communityBase = "https://huggingface.co/OwensLab/commfor-model-384/resolve/$communityRevision"
Get-VerifiedAsset `
    -Url "$communityBase/model.safetensors" `
    -Target (Join-Path $assetDirectory "commfor-model-384.safetensors") `
    -ExpectedSha256 "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387"
Get-VerifiedAsset `
    -Url "$communityBase/config.json" `
    -Target (Join-Path $assetDirectory "commfor-model-384.config.json") `
    -ExpectedSha256 "877416e73aac0fdbc1be723f5ddf674e78a2350305865ad9031564b287b60147"
