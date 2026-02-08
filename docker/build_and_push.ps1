# build and push the custom vllm worker image to dockerhub.
#
# usage:
#   .\docker\build_and_push.ps1 -DockerUser yourusername
#   .\docker\build_and_push.ps1 -DockerUser yourusername -Tag v1.0

param(
    [Parameter(Mandatory=$true)]
    [string]$DockerUser,

    [string]$ImageName = "engram-vllm-worker",

    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"

$fullImage = "${DockerUser}/${ImageName}:${Tag}"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  engram vllm worker image builder" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  output:  $fullImage"
Write-Host ""

# check docker is available
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "error: docker not found. install docker desktop first." -ForegroundColor Red
    exit 1
}

# build
Write-Host "building image..." -ForegroundColor Yellow
docker build `
    -t $fullImage `
    -f docker/vllm-worker/Dockerfile `
    docker/vllm-worker/

if ($LASTEXITCODE -ne 0) {
    Write-Host "build failed!" -ForegroundColor Red
    exit 1
}

Write-Host "build successful!" -ForegroundColor Green
Write-Host ""

# push
Write-Host "pushing to dockerhub..." -ForegroundColor Yellow
docker push $fullImage

if ($LASTEXITCODE -ne 0) {
    Write-Host "push failed! make sure you're logged in: docker login" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  pushed: $fullImage" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  use this image in deploy_vllm.py:"
Write-Host "    python scripts/deploy_vllm.py --image $fullImage"
Write-Host ""
