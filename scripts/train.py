from checkpoint import save_checkpoint
import matplotlib.pyplot as plt

def train_one_epoch(model, optimizer, device, epoch, total_epochs, dataloader):
    model.train()
    epoch_loss = []

    for x in dataloader:
        x = x.to(device)
        rand_mask = model.mask(x.shape[0], device)
        inp, rec = model(x, active_b1ff=rand_mask)
        _, patch_loss = model.forward_loss(inp, rec, rand_mask)

        anatomask_mask, _ = model.generate_mask(patch_loss, epoch=epoch, total_epoch=total_epochs)
        inp2, rec2 = model(x, active_b1ff=anatomask_mask)
        loss, _ = model.forward_loss(inp2, rec2, anatomask_mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    return sum(epoch_loss) / len(epoch_loss)

def train(model, optimizer, device, epochs, dataloader, save_path=None):
    loss_history = []
    for epoch in range(epochs):
        loss = train_one_epoch(model, optimizer, device, epoch, epochs, dataloader)
        # loss = 1 调试用
        loss_history.append(loss)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss:.6f}")
        if save_path:
            save_checkpoint(model, optimizer, epoch, save_path)
            print(f"✅ 模型已保存: {save_path}")
    plt.plot(loss_history, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()