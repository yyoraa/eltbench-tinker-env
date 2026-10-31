# Deployment notes

Fixes needed to run official ELT-Bench ingestion with rootless Docker. Neither
is in the upstream repository.

## REST API pagination

`elt-docker/rest_api/app.py` ignores `limit` and `offset`, while the manifest
paginates with `OffsetIncrement` (page size 9973). Any table with at least 9973
rows therefore never terminates. Apply after `app = Flask(__name__)`:

```python
@app.after_request
def honor_limit_offset(response):
    if 'limit' in request.args and response.is_json:
        data = response.get_json()
        if isinstance(data, list):
            offset = int(request.args.get('offset', 0))
            limit = int(request.args['limit'])
            response.set_data(jsonify(data[offset:offset + limit]).get_data())
    return response
```

## CPU limits under rootless Docker

Without CPU cgroup delegation, `docker run --cpus=2` fails with `NanoCPUs can
not be set`, so every sync exits with code 125. Either enable delegation (needs
root), or wrap `/usr/bin/docker` in the `airbyte-worker` container to drop
`--cpus=*`. The wrapper is lost if the container is recreated.

## Sandbox network

`DockerSandboxFactory` attaches sandboxes to `airbyte-compose-06311_airbyte_public`, where
the only other member is `airbyte-proxy`; `airbyte-server` and `airbyte-db` sit on the
internal network. Snowflake is reached over the bridge's normal outbound route. Egress is not
filtered. Airbyte's server URL inside the sandbox is `http://airbyte-proxy:8000`.

## Other notes

- LocalStack data does not survive a restart; reload the S3 objects afterwards.
- Select only the task's streams on the API connection. The custom source lists
  193 streams across all tasks.
- Rootless data lives under the Docker `data-root`; check disk quota first.
