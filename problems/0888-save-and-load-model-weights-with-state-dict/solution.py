import torch
import io

def copy_weights(src, dst):
    """
    Copy weights from src model to dst model via save/load round-trip.
    
    Args:
        src: Source model (nn.Module)
        dst: Destination model (nn.Module) - same architecture
    """
    # 1. Создаем буфер в памяти
    buffer = io.BytesIO()
    
    # 2. Сохраняем state_dict из src в буфер
    torch.save(src.state_dict(), buffer)
    
    # 3. Перемещаем курсор в начало буфера (для чтения)
    buffer.seek(0)
    
    # 4. Загружаем state_dict из буфера
    loaded_state = torch.load(buffer)
    
    # 5. Загружаем state_dict в dst (перезаписываем веса)
    dst.load_state_dict(loaded_state)