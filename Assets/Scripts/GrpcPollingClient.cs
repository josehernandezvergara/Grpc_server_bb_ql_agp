using UnityEngine;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.Threading;
using Grpc.Core;
using Warehouse; // namespace del código generado por .proto

public class GrpcPollingClient : MonoBehaviour
{
    [Header("Network")]
    public string target = "127.0.0.1:50051";
    public int pollIntervalMs = 500;

    [Header("Prefabs")]
    public GameObject agentPrefab;
    public GameObject boxPrefab;

    Channel channel;
    WarehouseService.WarehouseServiceClient client;
    CancellationTokenSource cts;
    ConcurrentQueue<CoordsResponse> responsesQueue = new ConcurrentQueue<CoordsResponse>();
    Dictionary<string, GameObject> entities = new Dictionary<string, GameObject>();

    async void Start()
    {
        Debug.Log("GrpcPollingClient: iniciando...");
        cts = new CancellationTokenSource();
        // start background polling
        _ = Task.Run(() => PollLoop(cts.Token));
    }

    async Task PollLoop(CancellationToken token)
    {
        int attempt = 0;
        while (!token.IsCancellationRequested)
        {
            try
            {
                if (channel == null) CreateChannel();
                // blocking call in background thread
                var reply = await Task.Run(() => client.GetCoords(new Empty()), token);
                responsesQueue.Enqueue(reply);
                attempt = 0;
            }
            catch (RpcException rex)
            {
                Debug.LogWarning($"gRPC RpcException: {rex.Status}. Reintentando...");
                attempt++;
                await Task.Delay(Mathf.Min(5000, 500 * attempt), token);
                DisposeChannel();
            }
            catch (TaskCanceledException) { break; }
            catch (System.Exception ex)
            {
                Debug.LogError("gRPC error: " + ex);
                attempt++;
                await Task.Delay(Mathf.Min(5000, 500 * attempt), token);
                DisposeChannel();
            }

            try { await Task.Delay(pollIntervalMs, token); } catch (TaskCanceledException) { break; }
        }
    }

    void CreateChannel()
    {
        try
        {
            channel = new Channel(target, ChannelCredentials.Insecure);
            client = new WarehouseService.WarehouseServiceClient(channel);
            Debug.Log("GrpcPollingClient: canal creado -> " + target);
        }
        catch (System.Exception e)
        {
            Debug.LogWarning("CreateChannel failed: " + e);
            DisposeChannel();
        }
    }

    void DisposeChannel()
    {
        try { channel?.ShutdownAsync().Wait(500); } catch { }
        channel = null; client = null;
    }

    void Update()
    {
        while (responsesQueue.TryDequeue(out var resp))
        {
            ApplyCoordsResponse(resp);
        }
    }

    void ApplyCoordsResponse(CoordsResponse resp)
    {
        var seen = new HashSet<string>();
        foreach (var obj in resp.Objects)
        {
            string id = obj.Id;
            seen.Add(id);
            Vector3 pos = new Vector3((float)obj.Position.X, (float)obj.Position.Y, (float)obj.Position.Z);

            if (!entities.TryGetValue(id, out var go) || go == null)
            {
                GameObject prefab = id.StartsWith("Agent") ? agentPrefab : boxPrefab;
                if (prefab == null) go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                else go = Instantiate(prefab, pos, Quaternion.identity);
                go.name = id;
                entities[id] = go;
            }
            entities[id].transform.position = Vector3.Lerp(entities[id].transform.position, pos, 0.65f);
        }

        // remove missing ones (optional)
        var toRemove = new List<string>();
        foreach (var k in entities.Keys)
            if (!seen.Contains(k)) toRemove.Add(k);
        foreach (var k in toRemove)
        {
            if (entities.TryGetValue(k, out var g)) Destroy(g);
            entities.Remove(k);
        }
    }

    async void OnApplicationQuit()
    {
        Debug.Log("GrpcPollingClient: cerrando...");
        cts?.Cancel();
        if (channel != null)
        {
            try { await channel.ShutdownAsync(); } catch { }
        }
    }

    void OnDestroy()
    {
        cts?.Cancel();
        DisposeChannel();
    }
}
