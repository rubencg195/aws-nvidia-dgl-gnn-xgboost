# OpenTofu configuration to create ECR repo and pull/tag/push NVIDIA container
resource "aws_ecr_repository" "financial_fraud_training" {
  name                 = local.ecr_repo_name
  image_tag_mutability = "MUTABLE"
  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = local.common_tags
}

# Pull from NGC and push to ECR using local-exec.
# Notes:
# - set +H disables history expansion to avoid `!: event not found`
# - StrictHostKeyChecking not used here (no SSH), but keep user rule in mind for future ssh
# - Reads NGC API key from nvidia_credentials.json (not committed; in .gitignore)
resource "null_resource" "pull_and_push_ngc_image" {
  depends_on = [aws_ecr_repository.financial_fraud_training]

  triggers = {
    repo         = aws_ecr_repository.financial_fraud_training.repository_url
    image_tag    = local.ecr_image_tag
    nv_image     = local.nvidia_image_full_name
    # re-run if credentials content hash changes
    cred_hash    = try(filesha256("${path.root}/nvidia_credentials.json"), "no-cred-file")
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-lc"]
    command = <<EOT
set -euo pipefail
set +H

if [ ! -f "${path.root}/nvidia_credentials.json" ]; then
  echo "nvidia_credentials.json not found at repo root" >&2
  exit 1
fi

# Read key via Python to avoid external jq dependency
NGC_API_KEY=$(python - <<'PY'
import json,sys
from pathlib import Path
p=Path(r"${path.root}")/"nvidia_credentials.json"
with p.open('r',encoding='utf-8') as f:
    print(json.load(f).get('ngc_api_key',''))
PY
)
if [ -z "$NGC_API_KEY" ]; then
  echo "ngc_api_key missing in nvidia_credentials.json" >&2
  exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=$(aws configure get region || echo ${data.aws_region.current.name})
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${aws_ecr_repository.financial_fraud_training.name}:${local.ecr_image_tag}"

# If the image tag already exists in ECR, skip to avoid long operations
echo "Checking if image exists in ECR: ${aws_ecr_repository.financial_fraud_training.name}:${local.ecr_image_tag}"
EXIST_DIGEST=$(aws ecr describe-images \
  --repository-name "${aws_ecr_repository.financial_fraud_training.name}" \
  --image-ids imageTag=${local.ecr_image_tag} \
  --query 'imageDetails[0].imageDigest' \
  --output text 2>/dev/null || echo "NONE")
if [ "$EXIST_DIGEST" != "None" ] && [ "$EXIST_DIGEST" != "NONE" ] && [ -n "$EXIST_DIGEST" ]; then
  echo "Image already exists in ECR (digest: $EXIST_DIGEST). Skipping pull/tag/push."
  exit 0
fi

echo "Logging into NGC..."
docker login nvcr.io --username '$oauthtoken' --password "$NGC_API_KEY"

echo "Checking if local image exists: ${local.nvidia_image_full_name}"
if ! docker image inspect ${local.nvidia_image_full_name} >/dev/null 2>&1; then
  echo "Pulling ${local.nvidia_image_full_name}"
  docker pull ${local.nvidia_image_full_name}
else
  echo "Local image already present: ${local.nvidia_image_full_name}"
fi

echo "Authenticating Docker to ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

echo "Tagging image"
docker tag ${local.nvidia_image_full_name} "$ECR_URI"

echo "Pushing to ECR"
docker push "$ECR_URI"

echo "Done: $ECR_URI"
EOT
  }
}
